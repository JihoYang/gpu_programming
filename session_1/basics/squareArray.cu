// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include <cuda_runtime.h>
#include <iostream>
using namespace std;

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

// Square vector on GPU
__global__ void vecSqr(float *d_a, int n){
	for (int i=0; i < n; i++){
		float val = d_a[i];
		val = val*val;
		d_a[i] = val;	
	}
}

int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *a = new float[n];
	size_t nbytes = (size_t)(n)*sizeof(float);
    for(int i=0; i<n; i++) a[i] = i;
    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = a[i];
        val = val*val;
        a[i] = val;
    }
    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    // GPU computation
    // reinit data
    for(int i=0; i<n; i++) a[i] = i;
	// Memory allocation on GPU
	float *d_a = NULL;
	cudaMalloc(&d_a, nbytes);
	cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);
	// Launch kernel
	dim3 block = dim3(128, 1, 1);
	dim3 grid = dim3((n+block.x-1)/block.x, 1, 1);
	vecSqr <<<grid, block>>> (d_a, n);
	// Copy back to CPU
 	cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaFree(d_a); CUDA_CHECK;
    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    // free CPU arrays
    delete[] a;
}
