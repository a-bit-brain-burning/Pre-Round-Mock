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
    constexpr int group=14;
    const int dt=tile%nd, wt=N==784 ? tile/(nd*group) : tile/nd;
    const int ht=N==784 ? int(blockIdx.y)*group+tile/nd%group : int(blockIdx.y);
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
    constexpr int group=14, tasks=((out+depth-1)/depth)*nw*(N==784 ? group : 1);
    overlap<N,S,H,W><<<dim3((tasks+3)/4,N==784 ? (nh+group-1)/group : nh),dim3(32,4)>>>(input,output);
}

__device__ __forceinline__ float4 stream_four(const float4* address) {
    float4 v;
    asm("ld.global.cs.L2::256B.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "l"(address));
    return v;
}

__device__ __forceinline__ float4 across_width(float4 v,int source) {
    return make_float4(__shfl_sync(0xffffffff,v.x,source),__shfl_sync(0xffffffff,v.y,source),
                       __shfl_sync(0xffffffff,v.z,source),__shfl_sync(0xffffffff,v.w,source));
}

__device__ __forceinline__ void write_strip(const float4 (&v)[2],float* output,int oh,int ow,int dt,bool valid) {
    const int lane=threadIdx.x&7;
    const int group=threadIdx.x&24;
    const float back0=v[0].z+v[0].w, back1=v[1].z+v[1].w;
    const float previous0=__shfl_sync(0xffffffff,back0,group+((lane+7)&7));
    const float halo=__shfl_sync(0xffffffff,back1,group+6);
    const float previous1=__shfl_sync(0xffffffff,lane==7 ? back0 : back1,group+((lane+7)&7));
    const float next0=__shfl_sync(0xffffffff,lane==0 ? v[1].x : v[0].x,group+((lane+1)&7));
    const float next1=__shfl_sync(0xffffffff,v[1].x,group+(lane==5 ? 7 : ((lane+1)&7)));
    if (valid && ow<392) {
        float2* dest=reinterpret_cast<float2*>(output+(oh*392+ow)*392)+dt*14+lane;
        const float center0=(v[0].x+v[0].y)+v[0].z;
        dest[0]=make_float2(((lane==0 ? halo : previous0)+center0)*(1.0f/125),
                            (center0+(v[0].w+next0))*(1.0f/125));
        if (lane<6) {
            const float center1=(v[1].x+v[1].y)+v[1].z;
            dest[8]=make_float2((previous1+center1)*(1.0f/125),(center1+(v[1].w+next1))*(1.0f/125));
        }
    }
}

template<bool interior>
__device__ __forceinline__ void lane_volume(const float* input,float* output,int ht,int wt,int dt) {
    constexpr int height=14;
    const int lane=threadIdx.x, group=lane/8, part=lane%8;
    const int h0=ht*(2*height)-2, w0=(wt*6+2*group-1)*2;
    const int d[2]={dt*56+4*part,dt*56+4*(part<6 ? 8+part : part==6 ? -1 : 14)};
    float4 older[4][2],newer[4][2];
    const auto read=[&](int h,int w,int q) {
        return read_group<784,interior>(input,h,w,min(780,max(0,d[q])));
    };
    #pragma unroll
    for (int j=0;j<4;++j) {
        #pragma unroll
        for (int q=0;q<2;++q) {
            older[j][q]=add(read(h0,w0+j,q),read(h0+1,w0+j,q));
            newer[j][q]=add(read(h0+2,w0+j,q),read(h0+3,w0+j,q));
        }
    }
    #pragma unroll 1
    for (int i=0;i<height;++i) {
        const int h=h0+2*i+4;
        float4 first[2],second[2];
        #pragma unroll
        for (int q=0;q<2;++q) {
            float4 vertical[4];
            #pragma unroll
            for (int j=0;j<4;++j) {
                const float4 edge=read(h,w0+j,q);
                vertical[j]=add(add(older[j][q],newer[j][q]),edge);
                if (i+1<height) {
                    older[j][q]=newer[j][q];
                    newer[j][q]=add(edge,read(h+1,w0+j,q));
                }
            }
            const float4 a=add(vertical[0],vertical[1]), b=add(vertical[2],vertical[3]);
            const float4 left=across_width(b,(lane+24)%32);
            const float4 right=across_width(vertical[0],(lane+8)%32);
            first[q]=add(add(left,a),vertical[2]);
            second[q]=add(add(a,b),right);
            if (q==1 && unsigned(d[1])>=784)
                first[q]=second[q]=make_float4(0,0,0,0);
        }
        const int ow=wt*6+group*2-1, oh=ht*height+i;
        write_strip(first,output,oh,ow,dt,group!=0);
        write_strip(second,output,oh,ow+1,dt,group!=3);
    }
}

__global__ void lane_tiles(const float* input,float* output) {
    const int tile=blockIdx.x*blockDim.y+threadIdx.y, ht=blockIdx.y;
    if (tile>=66*14) return;
    const int wt=tile/14, dt=tile%14;
    if (wt>0 && wt<65 && ht>0 && ht<27) lane_volume<true>(input,output,ht,wt,dt);
    else lane_volume<false>(input,output,ht,wt,dt);
}

template<int N,int rows>
__global__ void disjoint(const float* __restrict__ input,float* __restrict__ output) {
    constexpr int out=(N+2)/3;
    __shared__ float4 plane[rows][N/4+2];
    const int oh=blockIdx.y;
    const int h0=oh*3-1;
    const int w0=int(blockIdx.x)*rows*3-1;
    const bool full=h0>=0 && h0+2<N && w0>=0 && w0+rows*3<=N;
    #pragma unroll
    for (int row=0;row<rows;++row) {
        const int w=w0+3*row;
        const int t=(int(threadIdx.x)+(int(blockIdx.x)+oh)*32)%(N/4);
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
        const float value=(line[d-1]+line[d]+line[d+1])*(1.0f/27);
        float* dst=output+(size_t(oh)*out+ow)*out+pos%out;
        asm("st.global.cs.f32 [%0], %1;" :: "l"(dst),"f"(value):"memory");
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
            if (D==192) { launch_overlap<192,2,2,3>(input,output); return; }
            if (D==784) { lane_tiles<<<dim3(231,28),dim3(32,4)>>>(input,output); return; }
        }
        if (D==224 && kernel_size==7 && stride==3 && padding==3) {
            launch_overlap<224,3,1,2>(input,output); return;
        }
        if (kernel_size==3 && stride==3 && padding==1) {
            if (D==512) { disjoint<512,8><<<dim3(22,171),128>>>(input,output); return; }
            if (D==1024) { disjoint<1024,8><<<dim3(43,342),256>>>(input,output); return; }
        }
    }
    const size_t pad=2*size_t(padding);
    if (H+pad<size_t(kernel_size) || W+pad<size_t(kernel_size) || D+pad<size_t(kernel_size)) return;
    const size_t oh=(H+pad-kernel_size)/stride+1, ow=(W+pad-kernel_size)/stride+1, od=(D+pad-kernel_size)/stride+1;
    const size_t n=oh*ow*od, blocks=(n+255)/256;
    direct<<<blocks<65535?blocks:65535,256>>>(input,output,kernel_size,stride,padding,H,W,D,ow,od,n);
}
