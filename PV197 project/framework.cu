#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "device_launch_parameters.h"

// galaxy is stored as cartesian coordinates of its stars, each dimmension
// is in separate array
struct sGalaxy {
	float* x;
	float* y;
	float* z;
};

//IMPLEMENTED PART BEGIN
#define N 20000
//kernel.cu start

//function for getting distance of two given stars
__device__ void getDistance(float x1, float y1, float z1, float x2, float y2, float z2, float* res) {
	*res = (x1 - x2)*(x1 - x2)
		+ (y1 - y2)*(y1 - y2)
		+ (z1 - z2)*(z1 - z2);
}

//GPU executed kernel
__global__ void kernelGPU(sGalaxy A, sGalaxy B, float * diff, int n) {

	//get number of this thread
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	float tmp = 0.0f;

	//if this thread is past the scope of galaxies, thread needs not to be executed (for better block counts there is small number of "dummy" threads that is executed for some specific values)
	if (i>n - 1) return;

	//alocating first set of stars to and from block memory
	extern __shared__ float blockMem[];

	blockMem[threadIdx.x * 6] = A.x[i];
	blockMem[threadIdx.x * 6 + 1] = A.y[i];
	blockMem[threadIdx.x * 6 + 2] = A.z[i];
	blockMem[threadIdx.x * 6 + 3] = B.x[i];
	blockMem[threadIdx.x * 6 + 4] = B.y[i];
	blockMem[threadIdx.x * 6 + 5] = B.z[i];

	float Axi = blockMem[threadIdx.x * 6];
	float Ayi = blockMem[threadIdx.x * 6 + 1];
	float Azi = blockMem[threadIdx.x * 6 + 2];
	float Bxi = blockMem[threadIdx.x * 6 + 3];
	float Byi = blockMem[threadIdx.x * 6 + 4];
	float Bzi = blockMem[threadIdx.x * 6 + 5];

	__syncthreads();

	//values of star distances for each set
	float da;
	float db;

	// for n < 1800 alternative implementation was used, standard algorithm was crashing on this size but values were not tested for performance so optimization wasnt necessary
	if (n < 1800) {
		for (int j = i + 1; j < n; j++) {
			getDistance(Axi, Ayi, Azi, A.x[j], A.y[j], A.z[j], &da);
			getDistance(Bxi, Byi, Bzi, B.x[j], B.y[j], B.z[j], &db);
			//standard formula for this calculation uses two sqrt() operation which are computationaly exhausting, therefore we used optimised version
			tmp += -2 * sqrt(da*db) + da + db;
		}
		diff[i] = tmp;
		return;
	}

	//check number of stars that thread will compare (if values in block are in the scope)
	int f;
	if (i + blockDim.x - 1 > n) {
		f = n - i;
	}
	else {
		f = blockDim.x;
	}

	//computation for this block
	for (int j = threadIdx.x + 1; j < f; j++) {
		getDistance(Axi, Ayi, Azi, blockMem[j * 6], blockMem[j * 6 + 1], blockMem[j * 6 + 2], &da);
		getDistance(Bxi, Byi, Bzi, blockMem[j * 6 + 3], blockMem[j * 6 + 4], blockMem[j * 6 + 5], &db);
		tmp += -2 * sqrt(da*db) + da + db;
	}

	//computation of rest of the blocks with shared memory preloading
	int x = 0;
	int sh = blockDim.x;
	for (int j = i + (blockDim.x - threadIdx.x); j < n; j++) {
		if (sh == blockDim.x) {
			x = j + threadIdx.x;
			__syncthreads();
			if (x < n) {
				blockMem[threadIdx.x * 6] = A.x[x];
				blockMem[threadIdx.x * 6 + 1] = A.y[x];
				blockMem[threadIdx.x * 6 + 2] = A.z[x];
				blockMem[threadIdx.x * 6 + 3] = B.x[x];
				blockMem[threadIdx.x * 6 + 4] = B.y[x];
				blockMem[threadIdx.x * 6 + 5] = B.z[x];
			}
			__syncthreads();
			sh = 0;
		}

		getDistance(Axi, Ayi, Azi, blockMem[sh * 6], blockMem[sh * 6 + 1], blockMem[sh * 6 + 2], &da);
		getDistance(Bxi, Byi, Bzi, blockMem[sh * 6 + 3], blockMem[sh * 6 + 4], blockMem[sh * 6 + 5], &db);
		tmp += -2 * sqrt(da*db) + da + db;
		sh += 1;
	}

	//assigning the result
	diff[i] = tmp;
}


float solveGPU(sGalaxy A, sGalaxy B, int n) {

	const int optimalThreadCount = 256; //optimal for tested machine
	float result = 0.0f;
	float * h_tmp = (float*)calloc(n, sizeof(float)); //host memory set pointer
	float * d_tmp; //device memory set pointer
	int blockCount;
	int threadPerBlockCount;

	cudaMalloc((void**)&d_tmp, sizeof(float)*n);
	cudaMemset(d_tmp, 0.0f, sizeof(float)*n);

	//set block count to give best result on tested machine
	if (n < optimalThreadCount) {
		blockCount = 1;
		threadPerBlockCount = n;
	}
	else {
		blockCount = n / optimalThreadCount + 1;
		threadPerBlockCount = optimalThreadCount;
	}

	//run kernel on GPU
	kernelGPU <<< blockCount, threadPerBlockCount, threadPerBlockCount * sizeof(float) * 6 >>> (A, B, d_tmp, n);

	//obtain results
	cudaMemcpy(h_tmp, d_tmp, sizeof(float)*n, cudaMemcpyDeviceToHost);

	//sum and return final number
	for (int i = 0; i < n - 1; i++) {
		result += h_tmp[i];
	}

	cudaFree((void*)d_tmp);

	return sqrt(1 / ((float)n*((float)n - 1)) * result);
}
///kernel.cu end
//IMPLEMENTED PART END

float solveCPU(sGalaxy A, sGalaxy B, int n) {

	float diff = 0.0f;
	for (int i = 0; i < n - 1; i++) {
		float tmp = 0.0f;
		for (int j = i + 1; j < n; j++) {
			float da = sqrt((A.x[i] - A.x[j])*(A.x[i] - A.x[j])
				+ (A.y[i] - A.y[j])*(A.y[i] - A.y[j])
				+ (A.z[i] - A.z[j])*(A.z[i] - A.z[j]));
			float db = sqrt((B.x[i] - B.x[j])*(B.x[i] - B.x[j])
				+ (B.y[i] - B.y[j])*(B.y[i] - B.y[j])
				+ (B.z[i] - B.z[j])*(B.z[i] - B.z[j]));
			//XXX for large galaxies, more precise version of sum should be implemented, not required in this example 
			tmp += (da - db) * (da - db);
		}
		diff += tmp;
	}

	return sqrt(1 / ((float)n*((float)n - 1)) * diff);
}


void generateGalaxies(sGalaxy A, sGalaxy B, int n) {
	for (int i = 0; i < n; i++) {
		// create star in A at random position first
		A.x[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		A.y[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		A.z[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		// create star in B near star A
		// in small probability, create more displaced star
		if ((float)rand() / (float)RAND_MAX < 0.01f) {
			B.x[i] = A.x[i] + 10.0f * (float)rand() / (float)RAND_MAX;
			B.y[i] = A.y[i] + 10.0f * (float)rand() / (float)RAND_MAX;
			B.z[i] = A.z[i] + 10.0f * (float)rand() / (float)RAND_MAX;
		}
		else {
			B.x[i] = A.x[i] + 1.0f * (float)rand() / (float)RAND_MAX;
			B.y[i] = A.y[i] + 1.0f * (float)rand() / (float)RAND_MAX;
			B.z[i] = A.z[i] + 1.0f * (float)rand() / (float)RAND_MAX;
		}
	}
}

int main(int argc, char **argv) {
	sGalaxy A, B;
	A.x = A.y = A.z = B.x = B.y = B.z = NULL;
	sGalaxy dA, dB;
	dA.x = dA.y = dA.z = dB.x = dB.y = dB.z = NULL;
	float diff_CPU, diff_GPU;

	// parse command line
	int device = 0;
	if (argc == 2)
		device = atoi(argv[1]);
	if (cudaSetDevice(device) != cudaSuccess) {
		fprintf(stderr, "Cannot set CUDA device!\n");
		exit(1);
	}
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	printf("Using device %d: \"%s\"\n", device, deviceProp.name);

	// create events for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate and set host memory
	A.x = (float*)malloc(N * sizeof(A.x[0]));
	A.y = (float*)malloc(N * sizeof(A.y[0]));
	A.z = (float*)malloc(N * sizeof(A.z[0]));
	B.x = (float*)malloc(N * sizeof(B.x[0]));
	B.y = (float*)malloc(N * sizeof(B.y[0]));
	B.z = (float*)malloc(N * sizeof(B.z[0]));
	generateGalaxies(A, B, N);

	// allocate and set device memory
	if (cudaMalloc((void**)&dA.x, N * sizeof(dA.x[0])) != cudaSuccess
		|| cudaMalloc((void**)&dA.y, N * sizeof(dA.y[0])) != cudaSuccess
		|| cudaMalloc((void**)&dA.z, N * sizeof(dA.z[0])) != cudaSuccess
		|| cudaMalloc((void**)&dB.x, N * sizeof(dB.x[0])) != cudaSuccess
		|| cudaMalloc((void**)&dB.y, N * sizeof(dB.y[0])) != cudaSuccess
		|| cudaMalloc((void**)&dB.z, N * sizeof(dB.z[0])) != cudaSuccess) {
		fprintf(stderr, "Device memory allocation error!\n");
		goto cleanup;
	}
	cudaMemcpy(dA.x, A.x, N * sizeof(dA.x[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dA.y, A.y, N * sizeof(dA.y[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dA.z, A.z, N * sizeof(dA.z[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dB.x, B.x, N * sizeof(dB.x[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dB.y, B.y, N * sizeof(dB.y[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dB.z, B.z, N * sizeof(dB.z[0]), cudaMemcpyHostToDevice);

	// solve on CPU
	printf("Solving on CPU...\n");
	cudaEventRecord(start, 0);
	diff_CPU = solveCPU(A, B, N);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("CPU performance: %f megapairs/s\n",
		float(N)*float(N - 1) / 2.0f / time / 1e3f);

	// solve on GPU
	printf("Solving on GPU...\n");
	cudaEventRecord(start, 0);
	// run it 10x for more accurately timing results
	for (int i = 0; i < 10; i++)
		diff_GPU = solveGPU(dA, dB, N);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("GPU performance: %f megapairs/s\n",
		float(N)*float(N - 1) / 2.0f / time / 1e2f);

	printf("CPU diff: %f\nGPU diff: %f\n", diff_CPU, diff_GPU);
	// check GPU results
	if (fabsf((diff_CPU - diff_GPU) / ((diff_CPU + diff_GPU) / 2.0f)) < 0.01f)
		printf("Test OK :-).\n");
	else
		fprintf(stderr, "Data mismatch: %f should be %f :-(\n", diff_GPU, diff_CPU);

	getchar();
cleanup:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (dA.x) cudaFree(dA.x);
	if (dA.y) cudaFree(dA.y);
	if (dA.z) cudaFree(dA.z);
	if (dB.x) cudaFree(dB.x);
	if (dB.y) cudaFree(dB.y);
	if (dB.z) cudaFree(dB.z);
	if (A.x) free(A.x);
	if (A.y) free(A.y);
	if (A.z) free(A.z);
	if (B.x) free(B.x);
	if (B.y) free(B.y);
	if (B.z) free(B.z);

	return 0;
}
