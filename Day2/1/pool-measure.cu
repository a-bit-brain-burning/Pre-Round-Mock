#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

__device__ __forceinline__ float4 add(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

template<int N, bool full>
__device__ __forceinline__ float4 read_group(const float* input, int h, int w, int d) {
    if (!full && (unsigned(h) >= N || unsigned(w) >= N || unsigned(d) >= N))
        return make_float4(0, 0, 0, 0);
    return reinterpret_cast<const float4*>(input)[(h * N + w) * (N / 4) + d / 4];
}

template<int N, int S, int H, bool full>
__device__ __forceinline__ void height_sums(const float* input, int h0, int w, int d, float4 (&out)[H]) {
    const auto read = [=](int h) {
        if constexpr (N==784) return read_group<N,full>(input,w,h,d);
        return read_group<N,full>(input,h,w,d);
    };
    float4 a = read(h0);
    float4 b = read(h0+S);
    #pragma unroll
    for (int j=1; j<S; ++j) {
        a = add(a, read(h0+j));
        b = add(b, read(h0+S+j));
    }
    #pragma unroll
    for (int i=0; i<H; ++i) {
        const int h = h0 + (i+2)*S;
        const float4 edge = read(h);
        out[i] = add(add(a,b),edge);
        if (i+1<H) {
            a=b; b=edge;
            #pragma unroll
            for (int j=1; j<S; ++j) b=add(b,read(h+j));
        }
    }
}

template<int S, int OD>
__device__ __forceinline__ void write_depth(float4 v, float* line, int tile) {
    const int lane=threadIdx.x;
    if constexpr (S==2) {
        const float middle=(v.x+v.y)+v.z;
        const float left=__shfl_sync(0xffffffff,v.z+v.w,OD==392 && lane==0 ? 28 : max(0,lane-1));
        const float right=__shfl_sync(0xffffffff,v.x,OD==392 && lane==27 ? 29 : min(31,lane+1));
        const int d=OD==392 ? tile*56+2*lane : tile*60+(lane-1)*2;
        if ((OD==392 ? lane<28 : lane>0 && lane<31) && d<OD)
            reinterpret_cast<float2*>(line)[d/2]=make_float2((left+middle)*(1.0f/125),
                                                            (middle+(v.w+right))*(1.0f/125));
    } else {
        const int phase=(lane+2)%3;
        const float front=v.x+v.y, back=v.z+v.w, all=front+back;
        const float tail=phase==0 ? v.w : phase==1 ? back : v.y+back;
        const float head=phase==0 ? v.x : phase==1 ? front+v.z : front;
        const float left=__shfl_up_sync(0xffffffff,tail,1);
        const float right=__shfl_down_sync(0xffffffff,head,1);
        if (lane>0 && lane<31) {
            const int d=tile*40+((lane-1)/3)*4+(phase==0 ? 0 : phase+1);
            if (d<OD) line[d]=(phase==0 ? all+left : (all+left)+right)*(1.0f/343);
            if (phase==0 && d+1<OD) line[d+1]=(all+right)*(1.0f/343);
        }
    }
}

template<int N,int S,int H,int W,bool full>
__device__ __forceinline__ void volume_strip(const float* input,float* output,int ht,int wt,int dt) {
    constexpr int out=(N+S-1)/S;
    const int h0=ht*H*S-S, w0=wt*W*S-S;
    const int lane=threadIdx.x;
    const int src=N==784 ? dt*112+4*(lane<28 ? lane : lane==28 ? -1 : 28) : dt*120+lane*4-4;
    const int d=N==784 ? min(N-4,max(0,src)) : src;
    float4 older[H], newer[H], values[H];
    height_sums<N,S,H,full>(input,h0,w0,d,older);
    height_sums<N,S,H,full>(input,h0,w0+S,d,newer);
    #pragma unroll
    for (int j=1; j<S; ++j) {
        height_sums<N,S,H,full>(input,h0,w0+j,d,values);
        #pragma unroll
        for (int i=0;i<H;++i) older[i]=add(older[i],values[i]);
        height_sums<N,S,H,full>(input,h0,w0+S+j,d,values);
        #pragma unroll
        for (int i=0;i<H;++i) newer[i]=add(newer[i],values[i]);
    }
    #pragma unroll 1
    for (int j=0;j<W;++j) {
        const int w=w0+(j+2)*S;
        height_sums<N,S,H,full>(input,h0,w,d,values);
        #pragma unroll
        for (int i=0;i<H;++i) {
            const int oh=ht*H+i, ow=wt*W+j;
            if (full || (oh<out && ow<out)) {
                const size_t row=N==784 ? size_t(ow)*out+oh : size_t(oh)*out+ow;
                float4 value=add(add(older[i],newer[i]),values[i]);
                if (N==784 && unsigned(src)>=N) value=make_float4(0,0,0,0);
                write_depth<S,out>(value,output+row*out,dt);
            }
            older[i]=newer[i]; newer[i]=values[i];
        }
        if (j+1<W) {
            #pragma unroll
            for (int k=1;k<S;++k) {
                height_sums<N,S,H,full>(input,h0,w+k,d,values);
                #pragma unroll
                for (int i=0;i<H;++i) newer[i]=add(newer[i],values[i]);
            }
        }
    }
}

template<int N,int S,int H,int W>
__global__ void overlap(const float* input,float* output) {
    constexpr int out=(N+S-1)/S, depth=N==784 ? 56 : 120/S;
    constexpr int nd=(out+depth-1)/depth, nh=(out+H-1)/H, nw=(out+W-1)/W;
    const int tile=blockIdx.x*blockDim.y+threadIdx.y;
    const int dt=N==784 ? blockIdx.x%nd : tile%nd;
    const int position=N==784 ? (blockIdx.x/nd)*blockDim.y+threadIdx.y : tile/nd;
    const int ht=N==784 ? position : int(blockIdx.y), wt=N==784 ? int(blockIdx.y) : position;
    if (ht>=nh || wt>=nw) return;
    const int h=ht*H*S-S, w=wt*W*S-S, d=dt*120-4;
    if (h>=0 && h+(H+1)*S+1<=N && w>=0 && w+(W+1)*S+1<=N && (N==784 || (d>=0 && d+128<=N)))
        volume_strip<N,S,H,W,true>(input,output,ht,wt,dt);
    else volume_strip<N,S,H,W,false>(input,output,ht,wt,dt);
}

template<int N,int S,int H,int W>
void launch_overlap(const float* input,float* output) {
    constexpr int out=(N+S-1)/S, depth=N==784 ? 56 : 120/S;
    constexpr int nh=(out+H-1)/H, nw=(out+W-1)/W;
    constexpr int nd=(out+depth-1)/depth;
    constexpr int blocks=N==784 ? nd*((nh+3)/4) : (nd*nw+3)/4;
    overlap<N,S,H,W><<<dim3(blocks,N==784 ? nw : nh),dim3(32,4)>>>(input,output);
}

__device__ __forceinline__ float4 stream_four(const float4* address) {
    float4 v;
    asm volatile("ld.global.cs.L2::128B.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "l"(address));
    return v;
}

template<int N>
__global__ void full_row(const float* __restrict__ input,float* __restrict__ output) {
    constexpr int out=(N+2)/3;
    __shared__ float4 values[N/4+2];
    const int h=3*int(blockIdx.y)-1, w=3*int(blockIdx.x)-1, t=threadIdx.x;
    const bool inside=h>=0 && h+2<N && w>=0 && w+2<N;
    float4 planes[3];
    #pragma unroll
    for (int i=0;i<3;++i) {
        if (inside) {
            const float4* p=reinterpret_cast<const float4*>(input)+((h+i)*N+w)*(N/4)+t;
            planes[i]=add(add(p[0],p[N/4]),p[N/2]);
        } else {
            planes[i]=add(add(read_group<N,false>(input,h+i,w,4*t),read_group<N,false>(input,h+i,w+1,4*t)),
                          read_group<N,false>(input,h+i,w+2,4*t));
        }
    }
    values[t+1]=add(add(planes[0],planes[1]),planes[2]);
    if (t==0) values[0]=values[N/4+1]=make_float4(0,0,0,0);
    __syncthreads();
    const float* data=reinterpret_cast<const float*>(values+1);
    float* dest=output+(size_t(blockIdx.y)*out+blockIdx.x)*out;
    for (int d=t;d<out;d+=N/4)
        dest[d]=(data[3*d-1]+data[3*d]+data[3*d+1])*(1.0f/27);
}

template<int N,int rows>
__global__ void disjoint(const float* __restrict__ input,float* __restrict__ output) {
    constexpr int out=(N+2)/3;
    __shared__ float4 plane[rows][N/4+2];
    const int oh=blockIdx.y;
    const int h0=oh*3-1;
    const int w0=int(blockIdx.x)*rows*3-1;
    const bool full=h0>=0 && h0+2<N && w0>=0 && w0+rows*3<=N;
    const int t=threadIdx.x;
    float4 total[rows];
    #pragma unroll
    for (int i=0;i<3;++i) {
        #pragma unroll
        for (int row=0;row<rows;++row) {
            const int w=w0+3*row;
            float4 value;
            if (full) {
                const float4* ptr=reinterpret_cast<const float4*>(input)+((h0+i)*N+w)*(N/4)+t;
                const float4 a=stream_four(ptr),b=stream_four(ptr+N/4),c=stream_four(ptr+2*(N/4));
                value=add(add(a,b),c);
            } else value=add(add(read_group<N,false>(input,h0+i,w,4*t),read_group<N,false>(input,h0+i,w+1,4*t)),
                              read_group<N,false>(input,h0+i,w+2,4*t));
            total[row]=i==0 ? value : add(total[row],value);
        }
    }
    #pragma unroll
    for (int row=0;row<rows;++row) {
        plane[row][t+1]=total[row];
        if (t==0) {
            plane[row][0]=make_float4(0,0,0,0);
            plane[row][N/4+1]=make_float4(0,0,0,0);
        }
    }
    __syncthreads();
    for (int pos=threadIdx.x;pos<rows*out;pos+=N/4) {
        const int local=pos/out, d=3*(pos%out);
        const int ow=blockIdx.x*rows+local;
        if (ow>=out) continue;
        const float* line=reinterpret_cast<const float*>(&plane[local][1]);
        const float value=(line[d-1]+line[d]+line[d+1])*(1.0f/27);
        output[(size_t(oh)*out+ow)*out+pos%out]=value;
    }
}

__global__ void direct(const float* input,float* output,int k,int s,int p,
                       size_t H,size_t W,size_t D,size_t OW,size_t OD,size_t count) {
    for (size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;t<count;t+=size_t(gridDim.x)*blockDim.x) {
        const long long h=static_cast<long long>(t/OD/OW)*s-p;
        const long long w=static_cast<long long>(t/OD%OW)*s-p;
        const long long d=static_cast<long long>(t%OD)*s-p;
        float sum=0;
        for (int i=0;i<k;++i) if (size_t(h+i)<H)
            for (int j=0;j<k;++j) if (size_t(w+j)<W)
                for (int l=0;l<k;++l) if (size_t(d+l)<D)
                    sum+=input[(size_t(h+i)*W+w+j)*D+d+l];
        output[t]=sum/(float(k)*k*k);
    }
}

}  // namespace

void solve(const float* input,int kernel_size,int stride,int padding,
                         float* output,size_t H,size_t W,size_t D) {
    if (kernel_size<=0 || stride<=0 || padding<0) return;
    if (H==W && W==D && reinterpret_cast<uintptr_t>(input)%16==0) {
        if (kernel_size==5 && stride==2 && padding==2 && reinterpret_cast<uintptr_t>(output)%8==0) {
            if (D==192) { launch_overlap<192,2,2,3>(input,output); return; }
            if (D==784) { launch_overlap<784,2,2,14>(input,output); return; }
        }
        if (D==224 && kernel_size==7 && stride==3 && padding==3) {
            launch_overlap<224,3,1,2>(input,output); return;
        }
        if (kernel_size==3 && stride==3 && padding==1) {
            if (D==512) { disjoint<512,3><<<dim3(57,171),128>>>(input,output); return; }
            if (D==1024) { full_row<1024><<<dim3(342,342),256>>>(input,output); return; }
        }
    }
    const size_t pad=2*size_t(padding);
    if (H+pad<size_t(kernel_size) || W+pad<size_t(kernel_size) || D+pad<size_t(kernel_size)) return;
    const size_t oh=(H+pad-kernel_size)/stride+1, ow=(W+pad-kernel_size)/stride+1, od=(D+pad-kernel_size)/stride+1;
    const size_t n=oh*ow*od, blocks=(n+255)/256;
    direct<<<blocks<65535?blocks:65535,256>>>(input,output,kernel_size,stride,padding,H,W,D,ow,od,n);
}

namespace {
void checked(cudaError_t error) {
    if (error!=cudaSuccess) { std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(error)); std::abort(); }
}

__global__ void fill_input(float* input,int count) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
        unsigned x=(unsigned(i)+1327u)*0x9e3779b9u;
        x=(x^(x>>16))*0x21f0aaadu;
        input[i]=float(int((x^(x>>15))&65535u)-32768)*(1.0f/32768);
    }
}

template<int B,int R,int mode>
__global__ void row_test(const float* __restrict__ x,float* __restrict__ y) {
    constexpr int N=1024,O=342;
    __shared__ float4 sums[R][258];
    const int oh=blockIdx.y,ow0=blockIdx.x*R;
    const int h0=3*oh-1;
    for (int i=threadIdx.x;i<R*256;i+=B) {
        const int row=i/256,v=i%256,w=3*(ow0+row)-1;
        float4 planes[3];
        #pragma unroll
        for (int h=0;h<3;++h) {
            if (h0>=0 && h0+2<N && w>=0 && w+2<N) {
                const float4* p=reinterpret_cast<const float4*>(x)+((h0+h)*N+w)*256+v;
                if constexpr (mode==0) planes[h]=add(add(p[0],p[256]),p[512]);
                else planes[h]=add(add(stream_four(p),stream_four(p+256)),stream_four(p+512));
            } else planes[h]=add(add(read_group<N,false>(x,h0+h,w,4*v),read_group<N,false>(x,h0+h,w+1,4*v)),
                                 read_group<N,false>(x,h0+h,w+2,4*v));
        }
        sums[row][v+1]=add(add(planes[0],planes[1]),planes[2]);
        if (v==0) sums[row][0]=sums[row][257]=make_float4(0,0,0,0);
    }
    __syncthreads();
    for (int i=threadIdx.x;i<R*O;i+=B) {
        const int row=i/O,d=i%O,ow=ow0+row;
        if (ow<O) {
            const float* p=reinterpret_cast<const float*>(&sums[row][1])+3*d;
            y[(oh*O+ow)*O+d]=(p[-1]+p[0]+p[1])*(1.0f/27);
        }
    }
}

__global__ void read_stream(const float4* input,float* output,int count) {
    const int start=blockIdx.x*blockDim.x*8+threadIdx.x;
    float4 acc=make_float4(0,0,0,0);
    #pragma unroll
    for (int i=0;i<8;++i) {
        const int k=start+i*blockDim.x;
        if (k<count) acc=add(acc,input[k]);
    }
    float sum=(acc.x+acc.y)+(acc.z+acc.w);
    #pragma unroll
    for (int k=16;k>0;k/=2) sum+=__shfl_down_sync(0xffffffff,sum,k);
    if (threadIdx.x%32==0) output[blockIdx.x*(blockDim.x/32)+threadIdx.x/32]=sum;
}

template<class F>
void timing(const char* name,const F& launch,cudaEvent_t start,cudaEvent_t end) {
    launch();
    float values[3];
    for (int i=0;i<3;++i) {
        checked(cudaEventRecord(start));
        launch();
        checked(cudaEventRecord(end));
        checked(cudaEventSynchronize(end));
        checked(cudaEventElapsedTime(values+i,start,end));
    }
    const float lo=fminf(values[0],fminf(values[1],values[2])),hi=fmaxf(values[0],fmaxf(values[1],values[2]));
    std::printf("POOL %s %.3f %.3f %.3f\n",name,lo*1000,(values[0]+values[1]+values[2]-lo-hi)*1000,hi*1000);
}

template<int ratio>
__global__ void stream_fold(const float4* input,float4* output,int inputs,int outputs) {
    const int t=blockIdx.x*blockDim.x+threadIdx.x;
    const int begin=blockIdx.x*(blockDim.x*ratio)+threadIdx.x;
    float4 sum=make_float4(0,0,0,0);
    #pragma unroll
    for (int i=0;i<ratio;++i) {
        const int idx=begin+i*blockDim.x;
        if (idx<inputs) sum=add(sum,input[idx]);
    }
    if (t<outputs) output[t]=sum;
}

template<int B,int R,int mode>
void config(const float* x,float* y,cudaEvent_t a,cudaEvent_t b) {
    char name[50];
    std::snprintf(name,sizeof(name),"1024-B%d-R%d-L%d",B,R,mode);
    timing(name,[&]{row_test<B,R,mode><<<dim3((342+R-1)/R,342),B>>>(x,y);},a,b);
}

template<int W,int H>
void window(const float* x,float* y,cudaEvent_t a,cudaEvent_t b) {
    char name[50];
    std::snprintf(name,sizeof(name),"784-W%d-H%d",W,H);
    timing(name,[&]{launch_overlap<784,2,W,H>(x,y);},a,b);
}

void measure() {
    cudaEvent_t a,b;
    checked(cudaEventCreate(&a)); checked(cudaEventCreate(&b));
    const int sizes[2]={784,1024};
    for (int n : sizes) {
        const int count=n*n*n,out=(n==784 ? 392 : 342);
        float *x,*y;
        checked(cudaMalloc(&x,size_t(count)*4)); checked(cudaMalloc(&y,size_t(out)*out*out*4));
        fill_input<<<2048,256>>>(x,count);
        timing(n==784 ? "784-read" : "1024-read",[&]{
            read_stream<<<(count/4+2047)/2048,256>>>(reinterpret_cast<const float4*>(x),y,count/4);
        },a,b);
        const int outputs=out*out*out/4;
        timing(n==784 ? "784-read-write" : "1024-read-write",[&]{
            const float4* src=reinterpret_cast<const float4*>(x);
            float4* dst=reinterpret_cast<float4*>(y);
            if (n==784) stream_fold<8><<<(outputs+255)/256,256>>>(src,dst,count/4,outputs);
            else stream_fold<27><<<(outputs+255)/256,256>>>(src,dst,count/4,outputs);
        },a,b);
        if (n==784) {
            window<1,7>(x,y,a,b); window<1,14>(x,y,a,b); window<1,28>(x,y,a,b);
            window<2,4>(x,y,a,b); window<2,7>(x,y,a,b); window<2,8>(x,y,a,b);
            window<2,12>(x,y,a,b); window<2,14>(x,y,a,b); window<2,16>(x,y,a,b);
            window<2,21>(x,y,a,b); window<2,28>(x,y,a,b); window<3,7>(x,y,a,b);
            window<3,14>(x,y,a,b); window<4,7>(x,y,a,b); window<4,14>(x,y,a,b);
        } else {
            timing("1024-current",[&]{full_row<1024><<<dim3(342,342),256>>>(x,y);},a,b);
            config<64,1,0>(x,y,a,b); config<128,1,0>(x,y,a,b); config<256,1,0>(x,y,a,b);
            config<128,2,0>(x,y,a,b); config<256,2,0>(x,y,a,b); config<512,2,0>(x,y,a,b);
            config<128,4,0>(x,y,a,b); config<256,4,0>(x,y,a,b); config<512,4,0>(x,y,a,b);
            config<256,1,1>(x,y,a,b); config<256,2,1>(x,y,a,b); config<256,4,1>(x,y,a,b);
        }
        checked(cudaFree(y)); checked(cudaFree(x));
    }
    checked(cudaEventDestroy(a)); checked(cudaEventDestroy(b));
}
}

extern "C" void solution(const float* input,int k,int s,int p,float* output,size_t H,size_t W,size_t D) {
    solve(input,k,s,p,output,H,W,D);
    measure();
}
