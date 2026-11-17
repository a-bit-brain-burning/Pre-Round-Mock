#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace {

template<int N,int IH,int IW,int ID>
__device__ __forceinline__ void load_volume(const float* input,float* dest,int x,int y,int z) {
    constexpr int vectors=ID/4;
    for (int i=threadIdx.x;i<IH*IW*vectors;i+=blockDim.x) {
        const int h=z+i/(IW*vectors), w=y+i/vectors%IW, d=x+4*(i%vectors);
        const bool valid=unsigned(h)<N && unsigned(w)<N && unsigned(d)<N;
        const float* src=valid ? input+(size_t(h)*N+w)*N+d : input;
        const unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(dest+4*i));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(addr),"l"(src),"r"(valid ? 16 : 0):"memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
}

__device__ __forceinline__ float4 add(float4 a,float4 b) {
    return make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
}

template<int N,int H,int W,int D>
__global__ void three(const float* input,float* output) {
    constexpr int out=(N+2)/3, ih=3*H, iw=3*W, id=3*D+4;
    extern __shared__ __align__(128) float raw[];
    load_volume<N,ih,iw,id>(input,raw,int(blockIdx.x)*D*3-4,
                             int(blockIdx.y)*W*3-1,int(blockIdx.z)*H*3-1);
    for (int idx=threadIdx.x;idx<H*W*D;idx+=blockDim.x) {
        const int h=idx/(W*D), w=idx/D%W, d=idx%D;
        float sum=0;
        #pragma unroll
        for (int i=0;i<3;++i) {
            #pragma unroll
            for (int j=0;j<3;++j) {
                const float* p=raw+((3*h+i)*iw+3*w+j)*id+3*d+3;
                sum+=(p[0]+p[1])+p[2];
            }
        }
        const int oh=blockIdx.z*H+h, ow=blockIdx.y*W+w, od=blockIdx.x*D+d;
        if (oh<out && ow<out && od<out) output[(size_t(oh)*out+ow)*out+od]=sum*(1.0f/27);
    }
}

template<int S>
__device__ __forceinline__ void depth(float4 v,float* dest,int lane) {
    if constexpr (S==2) {
        const float middle=(v.x+v.y)+v.z;
        const float a=__shfl_up_sync(0xffffffff,v.z,1);
        const float b=__shfl_up_sync(0xffffffff,v.w,1);
        const float c=__shfl_down_sync(0xffffffff,v.x,1);
        if (lane>0 && lane<31)
            reinterpret_cast<float2*>(dest)[lane-1]=make_float2((a+b)+middle,middle+(v.w+c));
    } else {
        float sum=(v.x+v.y)+(v.z+v.w);
        #pragma unroll
        for (int distance=1;distance<32;distance*=2) {
            const float prev=__shfl_up_sync(0xffffffff,sum,distance);
            if (lane>=distance) sum+=prev;
        }
        float start=__shfl_up_sync(0xffffffff,sum,1);
        if (lane==0) start=0;
        const float p[4]={start+v.x,start+(v.x+v.y),start+((v.x+v.y)+v.z),sum};
        #pragma unroll
        for (int i=0;i<4;++i) {
            const int lo=3*i,hi=3*i+7;
            const float a=__shfl_sync(0xffffffff,p[lo%4],(3*lane+lo/4)%32);
            const float b=__shfl_sync(0xffffffff,p[hi%4],(3*lane+hi/4)%32);
            if (lane<10) dest[4*lane+i]=b-a;
        }
    }
}

template<int N,int S,int H,int W>
__global__ void overlapping(const float* input,float* output) {
    constexpr int out=(N+S-1)/S, k=2*S+1, ih=(H+1)*S+1, iw=(W+1)*S+1, id=128, od_tile=120/S;
    extern __shared__ __align__(128) float raw[];
    float* partial=raw+ih*iw*id;
    load_volume<N,ih,iw,id>(input,raw,int(blockIdx.x)*120-4,
                             int(blockIdx.y)*W*S-S,int(blockIdx.z)*H*S-S);
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const float4* packed=reinterpret_cast<const float4*>(raw);
    for (int w=warp;w<iw;w+=8) {
        const float4* p=packed+w*32+lane;
        float4 a=p[0], b=p[S*iw*32];
        #pragma unroll
        for (int i=1;i<S;++i) {
            a=add(a,p[i*iw*32]);
            b=add(b,p[(S+i)*iw*32]);
        }
        #pragma unroll
        for (int h=0;h<H;++h) {
            const int next=(h+2)*S;
            const float4 edge=p[next*iw*32];
            depth<S>(add(add(a,b),edge),partial+(h*iw+w)*od_tile,lane);
            if (h+1<H) {
                a=b; b=edge;
                #pragma unroll
                for (int i=1;i<S;++i) b=add(b,p[(next+i)*iw*32]);
            }
        }
    }
    __syncthreads();
    for (int idx=threadIdx.x;idx<H*W*od_tile;idx+=blockDim.x) {
        const int h=idx/(W*od_tile), w=idx/od_tile%W, d=idx%od_tile;
        const int oh=blockIdx.z*H+h, ow=blockIdx.y*W+w, od=blockIdx.x*od_tile+d;
        if (oh<out && ow<out && od<out) {
            float sum=0;
            #pragma unroll
            for (int j=0;j<k;++j) sum+=partial[(h*iw+w*S+j)*od_tile+d];
            output[(size_t(oh)*out+ow)*out+od]=sum*(1.0f/(k*k*k));
        }
    }
}

template<int N,int H,int W,int D>
void launch_three(const float* input,float* output) {
    constexpr int out=(N+2)/3, bytes=3*H*3*W*(3*D+4)*4;
    if (cudaFuncSetAttribute(three<N,H,W,D>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess) std::abort();
    three<N,H,W,D><<<dim3((out+D-1)/D,(out+W-1)/W,(out+H-1)/H),256,bytes>>>(input,output);
}

template<int N,int S,int H,int W>
void launch_overlap(const float* input,float* output) {
    constexpr int out=(N+S-1)/S, d=120/S, ih=(H+1)*S+1, iw=(W+1)*S+1;
    constexpr int bytes=(ih*iw*128+H*iw*d)*4;
    if (cudaFuncSetAttribute(overlapping<N,S,H,W>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)!=cudaSuccess) std::abort();
    overlapping<N,S,H,W><<<dim3((out+d-1)/d,(out+W-1)/W,(out+H-1)/H),256,bytes>>>(input,output);
}

__global__ void direct(const float* x,float* y,int k,int s,int p,size_t H,size_t W,size_t D,
                       size_t OW,size_t OD,size_t n) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        const long long h=static_cast<long long>(i/OD/OW)*s-p, w=static_cast<long long>(i/OD%OW)*s-p;
        const long long d=static_cast<long long>(i%OD)*s-p;
        float sum=0;
        for (int a=0;a<k;++a) if (size_t(h+a)<H)
            for (int b=0;b<k;++b) if (size_t(w+b)<W)
                for (int c=0;c<k;++c) if (size_t(d+c)<D) sum+=x[(size_t(h+a)*W+size_t(w+b))*D+size_t(d+c)];
        y[i]=sum/(float(k)*k*k);
    }
}
}

extern "C" void solution(const float* input,int kernel_size,int stride,int padding,float* output,size_t H,size_t W,size_t D) {
    if (kernel_size<=0 || stride<=0 || padding<0) return;
    if (H==W && W==D && reinterpret_cast<uintptr_t>(input)%16==0) {
        if (kernel_size==3 && stride==3 && padding==1) {
            if (D==512) { launch_three<512,2,6,32>(input,output); return; }
            if (D==1024) { launch_three<1024,2,6,32>(input,output); return; }
        }
        if (kernel_size==5 && stride==2 && padding==2) {
            if (D==192) { launch_overlap<192,2,2,4>(input,output); return; }
            if (D==784) { launch_overlap<784,2,4,6>(input,output); return; }
        }
        if (D==224 && kernel_size==7 && stride==3 && padding==3) {
            launch_overlap<224,3,2,3>(input,output); return;
        }
    }
    const size_t border=2*size_t(padding);
    if (H+border<size_t(kernel_size) || W+border<size_t(kernel_size) || D+border<size_t(kernel_size)) return;
    const size_t oh=(H+border-kernel_size)/stride+1,ow=(W+border-kernel_size)/stride+1,od=(D+border-kernel_size)/stride+1;
    const size_t n=oh*ow*od,blocks=(n+255)/256;
    direct<<<blocks<65535?blocks:65535,256>>>(input,output,kernel_size,stride,padding,H,W,D,ow,od,n);
}
