#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "GA_GPU.cuh"

__global__ void initRand(curandState* s, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &s[idx]);
}

__global__ void fitnessKernel(Individual* pop, const Point* pts, int popSize, int nPts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < popSize) {
        float c[5]; bool bad = false;
        for(int i=0; i<5; i++) {
            c[i] = __uint_as_float(pop[idx].genome[i]);
            if (!isfinite(c[i])) bad = true;
        }
        if (bad) { pop[idx].fitness = 1e10f; return; }

        float maxE = 0;
        for (int i = 0; i < nPts; i++) {
            float x = pts[i].x;
            float f = c[0] + c[1]*x + c[2]*x*x + c[3]*x*x*x + c[4]*x*x*x*x;
            maxE = fmaxf(maxE, fabsf(f - pts[i].y));
        }
        pop[idx].fitness = isfinite(maxE) ? maxE : 1e10f;
    }
}

__global__ void evolveKernel(Individual* pop, int popSize, curandState* s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + (popSize/2);
    if (idx < popSize) {
        int p1 = curand(&s[idx]) % (popSize/2);
        int p2 = curand(&s[idx]) % (popSize/2);
        int cp = curand(&s[idx]) % 159 + 1;
        for(int j=0; j<5; j++) {
            int shift = j*32;
            if (shift + 32 <= cp) pop[idx].genome[j] = pop[p1].genome[j];
            else if (shift >= cp) pop[idx].genome[j] = pop[p2].genome[j];
            else {
                unsigned int mask = (1U << (cp - shift)) - 1;
                pop[idx].genome[j] = (pop[p1].genome[j] & mask) | (pop[p2].genome[j] & ~mask);
            }
        }
        if (curand_uniform(&s[idx]) < 0.1f) pop[idx].genome[curand(&s[idx])%5] ^= (1U << (curand(&s[idx])%32));
    }
}

struct Comp { __host__ __device__ bool operator()(const Individual& a, const Individual& b) { return a.fitness < b.fitness; } };

extern "C" Individual runGA_GPU(const Point* h_pts, int nPts, int popSize, int gens) {
    Point *d_pts; Individual *d_pop; curandState *d_s;
    cudaMalloc(&d_pts, nPts*sizeof(Point));
    cudaMalloc(&d_pop, popSize*sizeof(Individual));
    cudaMalloc(&d_s, popSize*sizeof(curandState));
    cudaMemcpy(d_pts, h_pts, nPts*sizeof(Point), cudaMemcpyHostToDevice);

    std::vector<Individual> h_init(popSize);
    for(auto &ind : h_init) for(int i=0; i<5; i++) { float v = (rand()%200-100)/20.0f; memcpy(&ind.genome[i], &v, 4); }
    cudaMemcpy(d_pop, h_init.data(), popSize*sizeof(Individual), cudaMemcpyHostToDevice);

    int blk = (popSize + 255)/256;
    initRand<<<blk, 256>>>(d_s, time(0));

    for(int g=0; g<gens; g++) {
        fitnessKernel<<<blk, 256>>>(d_pop, d_pts, popSize, nPts);
        thrust::device_ptr<Individual> p(d_pop);
        thrust::sort(p, p + popSize, Comp());
        evolveKernel<<<blk, 256>>>(d_pop, popSize, d_s);
    }
    Individual res; cudaMemcpy(&res, d_pop, sizeof(Individual), cudaMemcpyDeviceToHost);
    cudaFree(d_pts); cudaFree(d_pop); cudaFree(d_s);
    return res;
}