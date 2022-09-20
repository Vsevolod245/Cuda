
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <math_constants.h>

#define N (256*256)

__global__ void kernel(float* data)
{
	int idx =(int)(blockIdx.x * blockDim.x + threadIdx.x);
	data[idx] = cosf(idx * CUDART_PI_F / (2.0 * N));
}

int main(int argc, char* argv[])
{
	float a[N];
	float* dev = NULL;
	cudaMalloc((void**)&dev, N * sizeof(float));
	kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);
	cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev);
	for (int idx = 0; idx < N; idx++)
		printf("Cosinus of %f = %.5f\n", 90.0 * idx / N, a[idx]);
	return 0;
}