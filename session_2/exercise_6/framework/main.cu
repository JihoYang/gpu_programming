// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

// Exercise 6

// Written by: Jiho Yang (M.Sc student in Computational Science & Engineering)
// Matriculation number: 03675799

#include "helper.h"
#include <iostream>
#include <string>
#include <unistd.h>
using namespace std;

const float pi = 3.141592653589793238462f;

__constant__ float kernel_constant[41 * 41 * sizeof(float)]; // Assumes r_max = 20
texture<float, 2, cudaReadModeElementType> texRef; // At file scope

// uncomment to use the camera
//#define CAMERA

// Convolution on texture memory
__global__ void convolution_texture(float *d_imgIn, float *d_imgOut, float *d_kernel, int w, int h, int nc, int w_kernel, int h_kernel, int r, bool kernel_is_const){
	// Get coordinates
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	// Kernel origin
	int mid = (w_kernel-1)/2;
	// Convolution
	for (size_t c = 0; c < nc; c++){
		size_t idx = x + (size_t)w * y + w * h * c;
		if (x < w && y < h){
			// Initialise output
			d_imgOut[idx] = 0;
			// Loop through kernel
			for (size_t j = 0; j < h_kernel; j++){
				for (size_t i = 0; i < w_kernel; i++){
					// Global kernel thread coordinate 
					int x_kernel_global = x - mid + i;
					int y_kernel_global = y - mid + j;
					// Kernel local index
					int idx_kernel_local = i + w_kernel*j;
					//
					float input = tex2D(texRef, x_kernel_global + 0.5f, y_kernel_global + 0.5f + h * c);
					if(kernel_is_const == true){
						d_imgOut[idx] += kernel_constant[idx_kernel_local] * input;
					} else{
						d_imgOut[idx] += d_kernel[idx_kernel_local] * input;
					}
				}
			}
		}
	__syncthreads();
	}
}

// Convolution on shared memory
__global__ void convolution_shared(float *d_imgIn, float *d_imgOut, float *d_kernel, int w, int h, int nc, int w_kernel, int h_kernel, int r, bool kernel_is_const){
	// Get coordinates
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int x_block_local = threadIdx.x;
	int y_block_local = threadIdx.y;
	// Set up shared memory dimensions
	int w_shared = blockDim.x + 2 * r;
	int h_shared = blockDim.y + 2 * r;
	// Create array in shared memory
	extern __shared__ float imgIn_shared[];	
	// Number of data loading for each thread - somewhat equivalent to number of blocks required to cover the whole shared memory
	int num_load = (w_shared * h_shared + (blockDim.x * blockDim.y - 1)) / (blockDim.x * blockDim.y);
	// Loop through channels
	for (size_t c = 0; c < nc; c++){
		// Fill in shared memory
		for (size_t i = 0; i < num_load; i++){	
			size_t idx_shared_local = x_block_local + (size_t)blockDim.x * y_block_local + i * blockDim.x * blockDim.y;
			// Get coordinates on shared memory
			int x_shared_local = idx_shared_local % w_shared;
			int y_shared_local = idx_shared_local / w_shared;
			// Get corresponding coordinates on global problem domain 
			int x_shared_global = x_shared_local + blockDim.x * blockIdx.x - r;
			int y_shared_global = y_shared_local + blockDim.y * blockIdx.y - r;
			// Clamping
			if (x_shared_global < 0){
				x_shared_global = 0;
			}
			if (x_shared_global > w - 1){
				x_shared_global = w - 1;
			}
			if (y_shared_global < 0){
				y_shared_global = 0;
			}
			if (y_shared_global > h - 1){
				y_shared_global = h - 1;
			}
			// Get global index of the local shared memory thread
			size_t idx_shared_global = x_shared_global + (size_t)w * y_shared_global + w * h * c;
			// 
			if (idx_shared_local < w_shared * h_shared){
				imgIn_shared[idx_shared_local] = d_imgIn[idx_shared_global];
			}
		}
		// Convolution
		size_t idx = x + (size_t)w * y + w * h * c;
		if (x < w && y < h){
			// Synchronise threads before applying convolution (make sure the shared memory is filled)
			__syncthreads();
			if (x < w && y < h){
				// Initialise output
				d_imgOut[idx] = 0;
				// Loop through kernel
				for (size_t j = 0; j < h_kernel; j++){
					for (size_t i = 0; i < w_kernel; i++){
						// Get coordinates of kernel in shared memory (note shared memory includes out of domain values)
						int x_block_shared = x_block_local + i;
						int y_block_shared = y_block_local + j;
						int idx_kernel_local = i + w_kernel * j;
						int idx_block_shared = x_block_shared + y_block_shared * w_shared;
						if (kernel_is_const == true){
							d_imgOut[idx] += kernel_constant[idx_kernel_local] * imgIn_shared[idx_block_shared];
						} else{
							d_imgOut[idx] += d_kernel[idx_kernel_local] * imgIn_shared[idx_block_shared];
						}
					}
				}
			}
		}
	}
}
	
// Convolution on global memory
__global__ void convolution_global(float *d_imgIn, float *d_imgOut, float *d_kernel, int w, int h, int nc, int w_kernel, int h_kernel, bool kernel_is_const){
	// Get coordinates
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	//int z = threadIdx.z + blockDim.z*blockIdx.z;
	// Get indices
	size_t idx = x + (size_t)w*y;
	// Initialise d_imgOut
	// Set origin
	int mid = (w_kernel-1)/2;
	// Convolution - Note x_kernel is the global x coordinate of kernel in the problem domain
	for (size_t c = 0; c < nc; c++){
		size_t idx_3d = idx + (size_t)w*h*c;
		d_imgOut[idx_3d] = 0.0f;
		if (x < w && y < h){
			for (size_t j = 0; j < h_kernel; j++){
				for (size_t i = 0; i < w_kernel; i++){
					// Boundary condition
					int x_kernel_global = x - mid + i;
					int y_kernel_global = y - mid + j;
					// clamping
					if (x_kernel_global < 0){
						x_kernel_global = 0;
					}
					if (x_kernel_global > w-1){
						x_kernel_global = w - 1;
					}
					if (y_kernel_global < 0){
						y_kernel_global = 0;
					}
					if (y_kernel_global > h - 1){
						y_kernel_global = h - 1;
					}
					// Get indices
					int idx_kernel_local = i + w_kernel*j;
					int idx_kernel_global = x_kernel_global + w*y_kernel_global + w*h*c;
					// Multiply and sum
					if (kernel_is_const == true){
						d_imgOut[idx_3d] += kernel_constant[idx_kernel_local] * d_imgIn[idx_kernel_global];
					} else{
						d_imgOut[idx_3d] += d_kernel[idx_kernel_local] * d_imgIn[idx_kernel_global];
					}
				}
			}
		}
	}
}

// Set up kernel
void get_kernel(float *kernel, int w_kernel, int h_kernel, const float pi, float sigma){
	//Set up parameters
	int origin = w_kernel/2;
	float total = 0.0f;
	// Define 2D Gaussian kernel
	for (size_t y_kernel = 0; y_kernel < h_kernel; y_kernel++){
		for (size_t x_kernel = 0; x_kernel < w_kernel; x_kernel++){
			int a = x_kernel - origin;
			int b = y_kernel - origin;
			int idx = x_kernel + w_kernel*y_kernel;
			kernel[idx] = (1.0f / (2.0f*pi*sigma*sigma))*exp(-1*((a*a+b*b) / (2*sigma*sigma)));
			total += kernel[idx];
		}
	}
	// Normalise kernel
	for (size_t y_kernel = 0; y_kernel < h_kernel; y_kernel++){
		for (size_t x_kernel = 0; x_kernel < w_kernel; x_kernel++){
			int idx = x_kernel + w_kernel*y_kernel;
			kernel[idx] /= total;
		}
	}
}

// main
int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;
	// Convolution kernel
	float sigma = 10.0f;
	getParam("sigma", sigma, argc, argv);
	cout << "sigma: " << sigma << endl;
    // ### Define your own parameters here as needed    
	bool kernel_is_const = false;
	getParam("kernel_is_const", kernel_is_const, argc, argv);
	cout << "Constant kernel memory : " << kernel_is_const << endl;

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
	// Define kernel dimensions
	int r = ceil(3*sigma);
	int w_kernel = r * 2 + 1;	  //windowing
	int h_kernel = w_kernel;  	  //Square kernel
	// Kernel information
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

	// Get array memory
	int nbytes = w * h * nc * sizeof(float);
	int nbytes_kernel = w_kernel * h_kernel * sizeof(float);
    // allocate raw input image array
    float *imgIn = new float[(size_t)nbytes];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);


    // ###
    // ###
    // ### TODO: Main computation
    // ###
    // ###

	// Kernel memory allocation
	float *kernel = new float[nbytes_kernel]; 
	// Create kernel
	get_kernel(kernel,  w_kernel, h_kernel, pi, sigma);
	// Processor type
	string processor;

	////////////////////////////////////////////////////////////////////// Block setting ///////////////////////////////////////////////////////////////////////

	dim3 block = dim3(128, 1, 1); 
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);

	////////////////////////////////////////////////////////////////////// Texture Memory ////////////////////////////////////////////////////////////////////// 

/*

	// Arrays
	float *d_kernel;
	float *d_imgIn;
	float *d_imgOut;
	// CUDA
    cudaMalloc(&d_kernel, nbytes_kernel);	CUDA_CHECK;
    cudaMalloc(&d_imgIn, nbytes); 			CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytes); 			CUDA_CHECK;
    cudaMemcpy(d_kernel, kernel, nbytes_kernel, cudaMemcpyHostToDevice);	CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);			    CUDA_CHECK;
	cudaMemcpyToSymbol(kernel_constant, kernel, nbytes_kernel);				CUDA_CHECK;
   	// Boundary condition
	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.addressMode[1] = cudaAddressModeClamp;
	texRef.filterMode = cudaFilterModeLinear;
	texRef.normalized = false;
	// Lecture note stuff..
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();							CUDA_CHECK;
	cudaBindTexture2D(NULL, &texRef, d_imgIn, &desc, w, nc * h, w * sizeof(d_imgIn[0]));	CUDA_CHECK;
	// Convolution
	Timer timer; timer.start();
	convolution_texture <<<grid, block>>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, w_kernel, h_kernel, r, kernel_is_const);
	timer.end();  float t = timer.get();
	// Lecture note stuff..
	cudaUnbindTexture(texRef);
    cudaMemcpy(imgOut, d_imgOut, nbytes, cudaMemcpyDeviceToHost); 										CUDA_CHECK;
 	// Free memory
    cudaFree(d_imgIn);  CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_kernel); CUDA_CHECK;
	// Type of processor
	processor = "GPU - texture memory";
	cout << processor << endl;
	cout << "time: " << t*1000 << " ms" << endl;

*/

	////////////////////////////////////////////////////////////////////// Shared Memory ////////////////////////////////////////////////////////////////////// 

/*

	// Arrays
	float *d_kernel;
	float *d_imgIn;
	float *d_imgOut;
	// CUDA
    cudaMalloc(&d_kernel, nbytes_kernel);	CUDA_CHECK;
    cudaMalloc(&d_imgIn, nbytes); 			CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytes); 			CUDA_CHECK;
    cudaMemcpy(d_kernel, kernel, nbytes_kernel, cudaMemcpyHostToDevice);	CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);			    CUDA_CHECK;
	cudaMemcpyToSymbol(kernel_constant, kernel, nbytes_kernel);				CUDA_CHECK;
	size_t smBytes = (block.x + 2 * r) * (block.y + 2 * r) * sizeof(float);
	// Convolution	
	Timer timer; timer.start();
    convolution_shared <<< grid, block, smBytes >>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, w_kernel, h_kernel, r, kernel_is_const);	CUDA_CHECK;
	timer.end();  float t = timer.get();
	cudaDeviceSynchronize(); 																			CUDA_CHECK;
    cudaMemcpy(imgOut, d_imgOut, nbytes, cudaMemcpyDeviceToHost); 										CUDA_CHECK;
 	// Free memory
    cudaFree(d_imgIn);  CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_kernel); CUDA_CHECK;
	// Type of processor
	processor = "GPU - shared memory";
	cout << processor << endl;
	cout << "time: " << t*1000 << " ms" << endl;

*/


	////////////////////////////////////////////////////////////////////// Global Memory ////////////////////////////////////////////////////////////////////// 



	// Arrays
	float *d_kernel;
	float *d_imgIn;
	float *d_imgOut;
	// CUDA
    cudaMalloc(&d_kernel, nbytes_kernel);	CUDA_CHECK;
    cudaMalloc(&d_imgIn, nbytes); 			CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytes); 			CUDA_CHECK;
    cudaMemcpy(d_kernel, kernel, nbytes_kernel, cudaMemcpyHostToDevice);	CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);			    CUDA_CHECK;
	cudaMemcpyToSymbol(kernel_constant, kernel, nbytes_kernel);				CUDA_CHECK;
	// Convolution
	Timer timer; timer.start();
    convolution_global <<< grid, block >>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, w_kernel, h_kernel, kernel_is_const);	CUDA_CHECK;
	timer.end();  float t = timer.get();
	cudaDeviceSynchronize(); 																								CUDA_CHECK;
    cudaMemcpy(imgOut, d_imgOut, nbytes, cudaMemcpyDeviceToHost); 															CUDA_CHECK;
 	// Free memory
    cudaFree(d_imgIn);  CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_kernel); CUDA_CHECK;
	// Type of processor
	processor = "GPU - global memory";
	cout << processor << endl;
	cout << "time: " << t*1000 << " ms" << endl;



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

	// free allocated arrays
#ifdef CAMERA
	delete[] imgIn;
	delete[] imgOut;
#else
	delete[] imgIn;
	delete[] imgOut;
	delete[] kernel;
#endif

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
