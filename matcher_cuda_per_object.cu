// matcher_cuda_per_object.cu : per-object kernel (device-side early exit)
#include <cstdio>
#include <cuda_runtime.h>
#include "matcher_api.h"

// ---------- kernel ----------
__global__ void k_check(
    const int* __restrict__ P, int N,
    const int* __restrict__ O, int M,
    int Hi, int Hj, double thr,
    int* __restrict__ foundFlag,
    int2* __restrict__ where)
{
    int pos = blockIdx.x;       // placement index
    if (pos >= Hi*Hj) return;

    // quick device-wide early exit
    if (atomicAdd(foundFlag, 0) != 0) return;

    int i = pos / Hj, j = pos % Hj;

    double sum = 0.0;
    int total = M*M;
    for(int t = threadIdx.x; t < total; t += blockDim.x){
        int y = t / M, x = t % M;
        int p = P[(i+y)*N + (j+x)];
        int o = O[y*M + x];
        sum += fabs(((double)p - (double)o) / (double)p);
    }

    extern __shared__ double s[];
    s[threadIdx.x] = sum;
    __syncthreads();

    for(int stride=blockDim.x>>1; stride>0; stride>>=1){
        if(threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x+stride];
        __syncthreads();
    }

    if(threadIdx.x==0 && s[0] < thr){
        if (atomicCAS(foundFlag,0,1)==0){
            where->x = i; where->y = j;
        }
    }
}

// ---------- public C-ABI ----------
extern "C" int matcher_exists_per_object_cuda(
    const int* hP, int N, int pic_id,
    const ObjView* obj,
    double threshold,
    int gpu_id,
    MatchResult* out)
{
    if(!hP || !obj || !obj->data || !out) return -1;
    int M = obj->M;
    int Hi = N - M + 1, Hj = N - M + 1;
    if(Hi<=0 || Hj<=0){ out->found=0; out->pic_id=pic_id; out->obj_id=obj->id; return 0; }

    int num_gpus=0; cudaGetDeviceCount(&num_gpus);
    if(num_gpus<=0) return -1;
    int dev = (gpu_id>=0? gpu_id:-gpu_id) % num_gpus;
    cudaSetDevice(dev);

    size_t bytesP = (size_t)N*(size_t)N*sizeof(int);
    size_t bytesO = (size_t)M*(size_t)M*sizeof(int);
    const int placements = Hi*Hj;

    int *dP=0,*dO=0,*dFound=0; int2* dWhere=0;
    cudaMalloc((void**)&dP, bytesP);
    cudaMalloc((void**)&dO, bytesO);
    cudaMalloc((void**)&dFound, sizeof(int));
    cudaMalloc((void**)&dWhere, sizeof(int2));
    cudaMemcpy(dP, hP, bytesP, cudaMemcpyHostToDevice);
    cudaMemcpy(dO, obj->data, bytesO, cudaMemcpyHostToDevice);
    cudaMemset(dFound, 0, sizeof(int));

    dim3 blocks(placements);
    dim3 threads(256);
    size_t shmem = threads.x*sizeof(double);
    k_check<<<blocks,threads,shmem>>>(dP,N,dO,M,Hi,Hj,threshold,dFound,dWhere);
    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess){ fprintf(stderr,"CUDA launch error: %s\n", cudaGetErrorString(e)); }

    int hFound=0; int2 hWhere={-1,-1};
    cudaMemcpy(&hFound, dFound, sizeof(int), cudaMemcpyDeviceToHost);
    if(hFound) cudaMemcpy(&hWhere, dWhere, sizeof(int2), cudaMemcpyDeviceToHost);

    cudaFree(dP); cudaFree(dO); cudaFree(dFound); cudaFree(dWhere);

    out->found = hFound;
    out->pic_id = pic_id;
    out->obj_id = obj->id;
    out->pos_i = hWhere.x;
    out->pos_j = hWhere.y;
    return 0;
}
