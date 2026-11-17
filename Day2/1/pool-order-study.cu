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
        constexpr int groups=OD==392 ? 28 : 24;
        const float left=__shfl_sync(0xffffffff,v.z+v.w,lane==0 ? groups : lane-1);
        const float right=__shfl_sync(0xffffffff,v.x,lane==groups-1 ? groups+1 : min(31,lane+1));
        const int d=tile*groups*2+2*lane;
        if (lane<groups && d<OD)
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
    constexpr int groups=N==784 ? 28 : 24;
    const int src=S==2 ? 4*(dt*groups+(lane<groups ? lane : lane==groups ? -1 : groups)) : dt*120+lane*4-4;
    const int d=min(N-4,max(0,src));
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
                if (unsigned(src)>=N) value=make_float4(0,0,0,0);
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
    constexpr int out=(N+S-1)/S, depth=N==784 ? 56 : N==192 ? 48 : 40;
    constexpr int nd=(out+depth-1)/depth, nh=(out+H-1)/H, nw=(out+W-1)/W;
    const int tile=blockIdx.x*blockDim.y+threadIdx.y;
    const int dt=N==784 ? blockIdx.x%nd : tile%nd;
    const int position=N==784 ? (blockIdx.x/nd)*blockDim.y+threadIdx.y : tile/nd;
    const int ht=N==784 ? position : int(blockIdx.y), wt=N==784 ? int(blockIdx.y) : position;
    if (ht>=nh || wt>=nw) return;
    const int h=ht*H*S-S, w=wt*W*S-S;
    if (h>=0 && h+(H+1)*S+1<=N && w>=0 && w+(W+1)*S+1<=N)
        volume_strip<N,S,H,W,true>(input,output,ht,wt,dt);
    else volume_strip<N,S,H,W,false>(input,output,ht,wt,dt);
}

template<int N,int S,int H,int W>
void launch_overlap(const float* input,float* output) {
    constexpr int out=(N+S-1)/S, depth=N==784 ? 56 : N==192 ? 48 : 40;
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

template<int N,int W=0,int H=0,bool height_first=false>
__global__ void __cluster_dims__(2,1,1) full_row(const float* __restrict__ input,float* __restrict__ output) {
    constexpr int out=(N+2)/3;
    __shared__ float4 values[N/4+2];
    int oh=blockIdx.y,ow=blockIdx.x;
    if constexpr(W>0) {
        static_assert(out%H==0 && out%W==0);
        const int id=blockIdx.x, local=id%(W*H), tile=id/(W*H);
        oh=tile/(out/W)*H+(height_first ? local%H : local/W);
        ow=tile%(out/W)*W+(height_first ? local/H : local%W);
    }
    const int h=3*oh-1, w=3*ow-1, t=threadIdx.x;
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
    float* dest=output+(size_t(oh)*out+ow)*out;
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

static void solve(const float* input,int kernel_size,int stride,int padding,
                         float* output,size_t H,size_t W,size_t D) {
    if (kernel_size<=0 || stride<=0 || padding<0) return;
    if (H==W && W==D && reinterpret_cast<uintptr_t>(input)%16==0) {
        if (kernel_size==5 && stride==2 && padding==2 && reinterpret_cast<uintptr_t>(output)%8==0) {
            if (D==192) { launch_overlap<192,2,2,4>(input,output); return; }
            if (D==784) { launch_overlap<784,2,2,4>(input,output); return; }
        }
        if (D==224 && kernel_size==7 && stride==3 && padding==3) {
            launch_overlap<224,3,2,3>(input,output); return;
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
template<int width,int height,int warps>
__global__ void grouped_window(const float* x,float* y) {
    constexpr int nh=196,nw=98,depth=7;
    static_assert(nh%width==0 && nw%height==0);
    const int work=blockIdx.x*warps+threadIdx.y;
    if (work>=nh*nw*depth) return;
    const int dt=work%depth, strip=work/depth, local=strip%(width*height), tile=strip/(width*height);
    const int ht=tile%(nh/width)*width+local%width, wt=tile/(nh/width)*height+local/width;
    const int h=4*ht-2, w=8*wt-2;
    if (h>=0 && h+7<=784 && w>=0 && w+11<=784)
        volume_strip<784,2,2,4,true>(x,y,ht,wt,dt);
    else volume_strip<784,2,2,4,false>(x,y,ht,wt,dt);
}
template<int W,int H,int warps>
void small_blocks(const float* x,float* y) {
    grouped_window<W,H,warps><<<(196*98*7+warps-1)/warps,dim3(32,warps)>>>(x,y);
}
template<int W,int H,bool turn=false>
void large_blocks(const float* x,float* y) {
    full_row<1024,W,H,turn><<<342*342,256>>>(x,y);
}
using Launch=void(*)(const float*,float*);
void default784(const float* x,float* y) { launch_overlap<784,2,2,4>(x,y); }
void default1024(const float* x,float* y) { full_row<1024><<<dim3(342,342),256>>>(x,y); }
struct Candidate { const char* name; Launch apply; };

void checked(cudaError_t e) {
    if (e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::abort(); }
}
__global__ void initialize(float* x,int count) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
        unsigned v=(unsigned(i)+3557)*0x9e3779b9u;
        v=(v^(v>>16))*0x21f0aaadu;
        x[i]=(int((v^(v>>15))&65535u)-32768)*(1.0f/32768);
    }
}
__global__ void verify(const float* a,const float* b,int count,int* bad) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count && a[i]!=b[i]) atomicExch(bad,1);
}
template<int N,int O,size_t count>
void measure(const Candidate (&choices)[count]) {
    constexpr int inputs=N*N*N,outputs=O*O*O;
    float *x,*y,*ref;
    int *bad,failed;
    cudaEvent_t a,b;
    checked(cudaEventCreate(&a)); checked(cudaEventCreate(&b));
    checked(cudaMalloc(&x,size_t(inputs)*4)); checked(cudaMalloc(&y,size_t(outputs)*4));
    checked(cudaMalloc(&ref,size_t(outputs)*4)); checked(cudaMalloc(&bad,sizeof(int)));
    checked(cudaMemset(bad,0,sizeof(int))); initialize<<<1024,256>>>(x,inputs);
    choices[0].apply(x,ref);
    for (auto choice:choices) {
        checked(cudaMemset(y,0xff,size_t(outputs)*4)); choice.apply(x,y);
        verify<<<(outputs+255)/256,256>>>(ref,y,outputs,bad);
    }
    checked(cudaMemcpy(&failed,bad,sizeof(int),cudaMemcpyDeviceToHost)); if (failed) std::abort();
    constexpr int repeats=7;
    float time[count][repeats];
    for (int rep=0;rep<repeats;++rep) for (size_t pos=0;pos<count;++pos) {
        const size_t i=(pos+rep)%count;
        checked(cudaEventRecord(a)); choices[i].apply(x,y); checked(cudaEventRecord(b));
        checked(cudaEventSynchronize(b)); checked(cudaEventElapsedTime(&time[i][rep],a,b));
    }
    for (size_t i=0;i<count;++i) {
        for (int a=0;a<repeats;++a) for (int b=a+1;b<repeats;++b) if (time[i][b]<time[i][a]) {
            const float tmp=time[i][a]; time[i][a]=time[i][b]; time[i][b]=tmp;
        }
        std::printf("ORDER %d %s min %.3f median %.3f max %.3f\n",N,choices[i].name,
                    time[i][0]*1000,time[i][repeats/2]*1000,time[i][repeats-1]*1000);
    }
    checked(cudaFree(bad)); checked(cudaFree(ref)); checked(cudaFree(y)); checked(cudaFree(x));
    checked(cudaEventDestroy(a)); checked(cudaEventDestroy(b));
}
}
extern "C" void solution(const float* x,int k,int s,int p,float* y,size_t h,size_t w,size_t d) {
    solve(x,k,s,p,y,h,w,d);
    const Candidate a[]={{"direct",default784},{"w4h2b4",small_blocks<4,2,4>},
        {"w4h2b7",small_blocks<4,2,7>},{"w4h2b8",small_blocks<4,2,8>},
        {"w14h7b7",small_blocks<14,7,7>},{"w49h7b7",small_blocks<49,7,7>},
        {"w196h1b7",small_blocks<196,1,7>},{"w1h98b7",small_blocks<1,98,7>}};
    const Candidate b[]={{"direct",default1024},{"w6h2",large_blocks<6,2>},
        {"w18h6",large_blocks<18,6>},{"w38h3",large_blocks<38,3>},
        {"w114h2",large_blocks<114,2>},{"h2w6",large_blocks<6,2,true>},
        {"h6w18",large_blocks<18,6,true>},{"h3w38",large_blocks<38,3,true>}};
    measure<784,392>(a); measure<1024,342>(b);
}
