#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define THREADS_PER_BLOCK 256


// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250
__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}

__global__ void NmDistanceKernel(
    const int pc0_n, const float *pc0_xyz, 
    const int pc1_n, const float *pc1_xyz, 
    const int *cls0, 
    int *histgram_translation,
    const float min_x, const float min_y, const float min_z,
    const float max_x, const float max_y, const float max_z, 
    const int window_x, const int window_y, const int window_z
    )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("thread and block params: blockIdx=%06d, threadIdx.x=%06d, tid=%06d \n", blockIdx.x, threadIdx.x, tid);

    float x0, y0, z0;
    int hist_idx;
    if (tid < pc0_n) 
    {
        x0 = pc0_xyz[tid * 3 + 0];
        y0 = pc0_xyz[tid * 3 + 1];
        z0 = pc0_xyz[tid * 3 + 2];
        hist_idx = cls0[tid * 1 + 0];
        // printf("tid=%6d, x0=%06f, y0=%06f, z0=%6f, hist_idx=%6f, \n", tid, x0, y0, z0, hist_idx);

    }
    else
    {
        x0 = 1e8f;
        y0 = 1e8f;
        z0 = 1e8f;

        hist_idx = 1e8;
    }

    __shared__ float shared_pc1[THREADS_PER_BLOCK * 3]; // pc1 labels are not needed

    for (int i = 0; i < pc1_n; i += THREADS_PER_BLOCK) 
    {
        // Copy a block of pc1 to shared memory
        int pc1_idx = i + threadIdx.x;
        // printf("shared pc1 i=%6d, threadIdx.x=%06d \n", i, threadIdx.x);

        if (pc1_idx < pc1_n) 
        {
            shared_pc1[threadIdx.x * 3 + 0] = pc1_xyz[pc1_idx * 3 + 0];
            shared_pc1[threadIdx.x * 3 + 1] = pc1_xyz[pc1_idx * 3 + 1];
            shared_pc1[threadIdx.x * 3 + 2] = pc1_xyz[pc1_idx * 3 + 2];
            // printf("shared pc1 i=%6d, threadIdx.x=%06d, x0=%06f, y0=%6f, z0=%6f, \n", i, threadIdx.x, shared_pc1[threadIdx.x * 3 + 0], shared_pc1[threadIdx.x * 3 + 1], shared_pc1[threadIdx.x * 3 + 2]);
        }

        __syncthreads();
            
        // Compute the distance between pc0[tid] and the points in shared_pc1
        int num_elems = min(THREADS_PER_BLOCK, pc1_n - i);
        // printf("num elems=%06d, i=%06d, pc1_n-i=%6d \n", num_elems, i, pc1_n-i);

        for (int j = 0; j < num_elems; j++) 
        {
            float x1 = shared_pc1[j * 3 + 0];
            float y1 = shared_pc1[j * 3 + 1];
            float z1 = shared_pc1[j * 3 + 2];

            // if (x0 < min_x || x0 >= max_x) continue;
            // if (y0 < min_y || y0 >= max_y) continue;
            // if (x1 < min_x || x1 >= max_x) continue;
            // if (y1 < min_y || y1 >= max_y) continue;
            float offset_x = (x1 - x0);
            float offset_y = (y1 - y0);
            float offset_z = (z1 - z0);

            if (offset_x >=min_x && offset_x < max_x && offset_y >= min_y && offset_y < max_y && offset_z >=0 && offset_z < max_z && tid<pc0_n) 
            {
                // printf("computer idx in window: hist_idx=%06d, x0=%06f, y0=%06f, z0=%06f, x1=%06f, y1=%06f, z1=%06f, ox=%06f, oy=%06f, oz=%6f \n", hist_idx, x0, y0, z0, x1, y1, z1, offset_x, offset_y, offset_z);
                // [): left included; right excluded.
                int p_x = __float2int_rd( (offset_x-min_x) / (max_x-min_x) * __int2float_rd(window_x) );
                int p_y = __float2int_rd( (offset_y-min_y) / (max_y-min_y) * __int2float_rd(window_y) );
                int p_z = __float2int_rd( (offset_z-min_z) / (max_z-min_z) * __int2float_rd(window_z) );

	            // histgram_translation[hist_idx*window_z*window_y*window_x+ p_z*window_y*window_x+p_y*window_x+p_x] += 1; 
                atomicAdd(histgram_translation + hist_idx*window_z*window_y*window_x+ p_z*window_y*window_x+p_y*window_x+p_x, 1);
                // printf("argmin: \
                //     tid=%06d, \
                //     i=%06d, j=%06d, \
                //     x0=%06f, y0=%06f, z0=%06f, \
                //     x1=%06f, y1=%06f, z1=%06f, \
                //     bin_x=%06d, bin_y=%06d, \
                //     argmin_pillar idx=%06d, \
                //     min_pillar/d=%06f, \
                //     argmin_pillar=%06d, j+i=%06d, prev_argmin=%6d. \n", 
                //     tid, 
                //     i, j, 
                //     x0, y0, z0,
                //     x1, y1, z1,
                //     bin_x, bin_y,
                //     tid*window_y*window_x+bin_y*window_x+bin_x,
                //     min_pillar[tid*window_y*window_x+bin_y*window_x+bin_x],
                //     argmin_pillar[tid*window_y*window_x+bin_y*window_x+bin_x], j+i, prev_argmin);
            }
        }
        // printf("Finish threadIdx.X=%06d done\n", threadIdx.x);
        __syncthreads(); // DO NOT REMOVE. It is needed.

    }
    // printf("Finish all threads \n");
}

int histgram_func(
    const at::Tensor &pc0, const at::Tensor &pc1, const at::Tensor &cls0,
    at::Tensor &histgram_translation, 
    const float min_x, const float min_y, const float min_z,
    const float max_x, const float max_y, const float max_z,
    const int window_x, const int window_y, const int window_z
    )
{
	at::cuda::CUDAGuard device_guard(pc0.device());
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	const int pc0_n = pc0.size(0);
	const int pc1_n = pc1.size(0);

	const int col_blocks_pc0 = DIVUP(pc0_n, THREADS_PER_BLOCK);
	dim3 blocks_pc0(col_blocks_pc0);
	const int col_blocks_pc1 = DIVUP(pc1_n, THREADS_PER_BLOCK);
	dim3 blocks_pc1(col_blocks_pc1);
	dim3 threads(THREADS_PER_BLOCK);
    // printf("threads config: col_blocks_pc0=%6d, col_blocks_pc1=%06d, \n", col_blocks_pc0, col_blocks_pc1);
	
    // printf("params: min_x=%6f, min_y=%06f, min_z=%06f, max_x=%06f, max_y=%06f, max_z=%06f, window_x=%06d, window_y=%06d, window_z=%06d \n", min_x, min_y, min_z, max_x, max_y, max_z, window_x, window_y, window_z);
	NmDistanceKernel<<<blocks_pc0, threads, 0, stream>>>(
        pc0_n, pc0.data_ptr<float>(), 
        pc1_n, pc1.data_ptr<float>(), 
        cls0.data_ptr<int>(), 
        histgram_translation.data_ptr<int>(), 
        min_x, min_y, min_z, 
        max_x, max_y, max_z, 
        window_x, window_y, window_z
    );
	
	AT_CUDA_CHECK(cudaGetLastError());

	return 1;
}
