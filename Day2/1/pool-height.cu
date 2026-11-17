#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

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
    float4 a = read_group<N, full>(input, h0, w, d);
    float4 b = read_group<N, full>(input, h0+S, w, d);
    #pragma unroll
    for (int j=1; j<S; ++j) {
        a = add(a, read_group<N,full>(input, h0+j, w, d));
        b = add(b, read_group<N,full>(input, h0+S+j, w, d));
    }
    #pragma unroll
    for (int i=0; i<H; ++i) {
        const int h = h0 + (i+2)*S;
        const float4 edge = read_group<N,full>(input, h, w, d);
        out[i] = add(add(a,b),edge);
        if (i+1<H) {
            a=b; b=edge;
            #pragma unroll
            for (int j=1; j<S; ++j) b=add(b,read_group<N,full>(input,h+j,w,d));
        }
    }
}

template<int S, int OD>
__device__ __forceinline__ void write_depth(float4 v, float* line, int tile) {
    const int lane=threadIdx.x;
    if constexpr (S==2) {
        const float middle=(v.x+v.y)+v.z;
        const float left2=__shfl_up_sync(0xffffffff,v.z,1);
        const float left=__shfl_up_sync(0xffffffff,v.w,1);
        const float right=__shfl_down_sync(0xffffffff,v.x,1);
        const int d=tile*60+(lane-1)*2;
        if (lane>0 && lane<31 && d<OD)
            reinterpret_cast<float2*>(line)[d/2]=make_float2(((left2+left)+middle)*(1.0f/125),
                                                            (middle+(v.w+right))*(1.0f/125));
    } else {
        float sum=(v.x+v.y)+(v.z+v.w);
        #pragma unroll
        for (int delta=1; delta<32; delta*=2) {
            const float prev=__shfl_up_sync(0xffffffff,sum,delta);
            if (lane>=delta) sum+=prev;
        }
        float start=__shfl_up_sync(0xffffffff,sum,1);
        if (lane==0) start=0;
        const float p[4]={start+v.x,start+(v.x+v.y),start+((v.x+v.y)+v.z),sum};
        #pragma unroll
        for (int i=0; i<4; ++i) {
            const int lo=3*i, hi=3*i+7;
            const float a=__shfl_sync(0xffffffff,p[lo%4],(3*lane+lo/4)%32);
            const float b=__shfl_sync(0xffffffff,p[hi%4],(3*lane+hi/4)%32);
            const int d=tile*40+lane*4+i;
            if (lane<10 && d<OD) line[d]=(b-a)*(1.0f/343);
        }
    }
}

template<int N,int S,int H,int W,bool full>
__device__ __forceinline__ void volume_strip(const float* input,float* output,int ht,int wt,int dt) {
    constexpr int out=(N+S-1)/S;
    const int h0=ht*H*S-S, w0=wt*W*S-S, d=dt*120+int(threadIdx.x)*4-4;
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
                write_depth<S,out>(add(add(older[i],newer[i]),values[i]),
                                   output+(size_t(oh)*out+ow)*out,dt);
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
    constexpr int out=(N+S-1)/S, depth=120/S;
    constexpr int nd=(out+depth-1)/depth, nw=(out+W-1)/W;
    const int tile=blockIdx.x*blockDim.y+threadIdx.y;
    if (tile>=nd*nw) return;
    const int wt=tile/nd, dt=tile%nd, ht=blockIdx.y;
    const int h=ht*H*S-S, w=wt*W*S-S, d=dt*120-4;
    if (h>=0 && h+(H+1)*S+1<=N && w>=0 && w+(W+1)*S+1<=N && d>=0 && d+128<=N)
        volume_strip<N,S,H,W,true>(input,output,ht,wt,dt);
    else volume_strip<N,S,H,W,false>(input,output,ht,wt,dt);
}

template<int N,int S,int H,int W>
void launch_overlap(const float* input,float* output) {
    constexpr int out=(N+S-1)/S, depth=120/S;
    constexpr int tasks=((out+depth-1)/depth)*((out+W-1)/W);
    overlap<N,S,H,W><<<dim3((tasks+3)/4,(out+H-1)/H),dim3(32,4)>>>(input,output);
}

__device__ __forceinline__ float4 stream_four(const float4* address) {
    float4 v;
    asm("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "l"(address));
    return v;
}

__device__ __forceinline__ float2 depth_pair(float4 values,int lane) {
    const float middle=(values.x+values.y)+values.z;
    const float left=__shfl_sync(0xffffffff,values.z+values.w,lane==0 ? 28 : lane-1);
    const float right=__shfl_sync(0xffffffff,values.x,lane==27 ? 29 : (lane+1)%32);
    return make_float2(left+middle,middle+(values.w+right));
}

template<bool interior>
__device__ __forceinline__ float4 load_strip(const float* input,int h,int w,int d,bool valid_depth) {
    if (!valid_depth || (!interior && (unsigned(h)>=784 || unsigned(w)>=784)))
        return make_float4(0,0,0,0);
    return reinterpret_cast<const float4*>(input)[(h*784+w)*196+d/4];
}

template<int H,int W,bool interior>
__device__ __forceinline__ void publish_pairs(const float* input,float4* partial,int h0,int w,int d,bool valid_depth) {
    const int warp=threadIdx.y, lane=threadIdx.x;
    float4 older[2],newer[2];
    #pragma unroll
    for (int j=0;j<2;++j) {
        older[j]=add(load_strip<interior>(input,h0,w+j,d,valid_depth),load_strip<interior>(input,h0+1,w+j,d,valid_depth));
        newer[j]=add(load_strip<interior>(input,h0+2,w+j,d,valid_depth),load_strip<interior>(input,h0+3,w+j,d,valid_depth));
    }
    #pragma unroll
    for (int oh=0;oh<H;++oh) {
        const int h=h0+2*oh+4;
        float4 height[2];
        #pragma unroll
        for (int j=0;j<2;++j) {
            const float4 edge=load_strip<interior>(input,h,w+j,d,valid_depth);
            height[j]=add(add(older[j],newer[j]),edge);
            if (oh+1<H) {
                older[j]=newer[j];
                newer[j]=add(edge,load_strip<interior>(input,h+1,w+j,d,valid_depth));
            }
        }
        const float2 pair=depth_pair(add(height[0],height[1]),lane);
        const float2 first=depth_pair(height[0],lane);
        partial[(oh*(W+2)+warp)*32+lane]=make_float4(pair.x,pair.y,first.x,first.y);
    }
}

template<int H,int W>
__global__ void bricks(const float* input,float* output) {
    extern __shared__ __align__(16) float4 partial[];
    const int lane=threadIdx.x, warp=threadIdx.y;
    const int h0=int(blockIdx.z)*2*H-2;
    const int w=(int(blockIdx.y)*W+warp-1)*2;
    const int column=lane<28 ? lane : lane==28 ? -1 : 28;
    const int d=int(blockIdx.x)*112+4*column;
    const bool active=lane<30 && unsigned(d)<784;
    if (h0>=0 && h0+2*H+2<784 && unsigned(w)<783)
        publish_pairs<H,W,true>(input,partial,h0,w,d,active);
    else publish_pairs<H,W,false>(input,partial,h0,w,d,active);
    __syncthreads();
    if (lane>=28 || warp==0 || warp==W+1) return;
    const int ow=int(blockIdx.y)*W+warp-1, od=int(blockIdx.x)*56+lane*2;
    #pragma unroll
    for (int i=0;i<H;++i) {
        const float4* p=partial+(i*(W+2)+warp)*32+lane;
        const float4 left=p[-32],center=p[0],right=p[32];
        const int oh=int(blockIdx.z)*H+i;
        float* dst=output+(oh*392+ow)*392+od;
        *reinterpret_cast<float2*>(dst)=make_float2(((left.x+center.x)+right.z)*(1.0f/125),
                                                    ((left.y+center.y)+right.w)*(1.0f/125));
    }
}

template<int N,int rows>
__global__ void disjoint(const float* __restrict__ input,float* __restrict__ output) {
    constexpr int out=(N+2)/3;
    __shared__ float4 plane[rows][N/4+2];
    const int h0=int(blockIdx.y)*3-1;
    const int w0=int(blockIdx.x)*rows*3-1;
    const bool full=h0>=0 && h0+2<N && w0>=0 && w0+rows*3<=N;
    #pragma unroll
    for (int row=0;row<rows;++row) {
        const int w=w0+3*row;
        const int t=threadIdx.x;
        float4 sums[3];
        if (full) {
            const float4* base=reinterpret_cast<const float4*>(input)+((h0*N+w)*(N/4)+t);
            #pragma unroll
            for (int i=0;i<3;++i) {
                const float4* ptr=base+i*(N*N/4);
                sums[i]=add(add(stream_four(ptr),stream_four(ptr+N/4)),stream_four(ptr+2*(N/4)));
            }
        } else {
            #pragma unroll
            for (int i=0;i<3;++i) {
                sums[i]=add(add(read_group<N,false>(input,h0+i,w,4*t),read_group<N,false>(input,h0+i,w+1,4*t)),
                            read_group<N,false>(input,h0+i,w+2,4*t));
            }
        }
        plane[row][t+1]=add(add(sums[0],sums[1]),sums[2]);
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
        output[(size_t(blockIdx.y)*out+ow)*out+pos%out]=(line[d-1]+line[d]+line[d+1])*(1.0f/27);
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

extern "C" void solution(const float* input,int kernel_size,int stride,int padding,
                         float* output,size_t H,size_t W,size_t D) {
    if (kernel_size<=0 || stride<=0 || padding<0) return;
    if (H==W && W==D && reinterpret_cast<uintptr_t>(input)%16==0) {
        if (kernel_size==5 && stride==2 && padding==2 && reinterpret_cast<uintptr_t>(output)%8==0) {
            if (D==192) { launch_overlap<192,2,1,3>(input,output); return; }
            if (D==784) {
                constexpr int h=14,w=7,bytes=h*(w+2)*32*sizeof(float4);
                cudaFuncSetAttribute(bricks<h,w>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
                bricks<h,w><<<dim3(7,392/w,392/h),dim3(32,w+2),bytes>>>(input,output);
                return;
            }
        }
        if (D==224 && kernel_size==7 && stride==3 && padding==3) {
            launch_overlap<224,3,1,2>(input,output); return;
        }
        if (kernel_size==3 && stride==3 && padding==1) {
            if (D==512) { disjoint<512,4><<<dim3(43,171),128>>>(input,output); return; }
            if (D==1024) { disjoint<1024,4><<<dim3(86,342),256>>>(input,output); return; }
        }
    }
    const size_t pad=2*size_t(padding);
    if (H+pad<size_t(kernel_size) || W+pad<size_t(kernel_size) || D+pad<size_t(kernel_size)) return;
    const size_t oh=(H+pad-kernel_size)/stride+1, ow=(W+pad-kernel_size)/stride+1, od=(D+pad-kernel_size)/stride+1;
    const size_t n=oh*ow*od, blocks=(n+255)/256;
    direct<<<blocks<65535?blocks:65535,256>>>(input,output,kernel_size,stride,padding,H,W,D,ow,od,n);
}
