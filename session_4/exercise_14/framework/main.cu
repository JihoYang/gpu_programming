// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

// Exercise 8

// Written by: Jiho Yang (M.Sc student in Computational Science & Engineering)
// Matriculation number: 03675799

#include "helper.h"
#include <iostream>
#include <string>
#include <unistd.h>
using namespace std;

const float pi = 3.141592653589793238462f;


// uncomment to use the camera
//#define CAMERA

// Compute eigenvalue of a 2 by 2 matrix
__device__ void compute_eigenvalue(float *d_eigen_value, float d_t1_val, float d_t2_val, float d_t3_val){
	// Define matrix	
	float A[4] = {d_t1_val, d_t2_val, d_t2_val, d_t3_val};
	// Define elements
	float a = A[0];
	float b = A[1];
	float c = A[2];
	float d = A[3];	
	// Trace and determinant
	float T = a + d;
	float D = a*d - b*c;
	// Compute eigenvalue
	d_eigen_value[0] = T/2 + sqrtf(T*T/4-D);
	d_eigen_value[1] = T/2 - sqrtf(T*T/4-D);
	// Sort eigenvalue array
	if (d_eigen_value[0] > d_eigen_value[1]){
		float swap = d_eigen_value[0];
		d_eigen_value[0] = d_eigen_value[1];
		d_eigen_value[1] = swap;
	}
}

// Feature Detection
__global__ void feature_detection(float *d_imgOut, float *d_eigen_value, float *d_t1, float *d_t2, float *d_t3, int w, int h, float alpha, float beta){
	// Get coordinates
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	// Get index
	int idx = x + (size_t)w*y;
	// Get eigenvalue
	compute_eigenvalue(d_eigen_value, d_t1[idx], d_t2[idx], d_t3[idx]);
	// Mark corner as red
	if (d_eigen_value[0] > alpha){
		d_imgOut[idx] = 1;
		d_imgOut[idx + (size_t)w*h] = 0;
		d_imgOut[idx + (size_t)w*h*2] = 0;
	} 
	// Mark edge as yellow
	else if (d_eigen_value[1] >= alpha && alpha >= beta && beta >= d_eigen_value[0]){
		d_imgOut[idx] = 1;
		d_imgOut[idx + (size_t)w*h] = 1;
		d_imgOut[idx + (size_t)w*h*2] = 0;
	} else {
		d_imgOut[idx] = d_imgOut[idx]*0.5;
		d_imgOut[idx + (size_t)w*h] = d_imgOut[idx + (size_t)w*h]*0.5;
		d_imgOut[idx + (size_t)w*h*2] = d_imgOut[idx + (size_t)w*h*2]*0.5;
	}
}


// Compute M
__global__ void compute_M(float *d_m1, float *d_m2, float *d_m3, float *d_gradx, float *d_grady, int w, int h, int nc){
	// Get coordinates
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	// Get index in matrices m
	size_t idx_2d = x + (size_t)w*y;
	// Initialise sums
	float sum1 = 0;	
	float sum2 = 0;
	float sum3 = 0;
	// Loop through channels
	for (size_t c = 0 ; c < nc; c++){
		// Get index
		size_t idx = x + (size_t)w*y + (size_t)w*h*c;	
		sum1 += d_gradx[idx] * d_gradx[idx];
		sum2 += d_gradx[idx] * d_grady[idx];
		sum3 += d_grady[idx] * d_grady[idx];
	}
	// Fill matrices
	d_m1[idx_2d] = sum1;
	d_m2[idx_2d] = sum2;
	d_m3[idx_2d] = sum3;
}

// Rotationally robust gradient
__global__ void rotational_gradient(float *d_imgIn, float *d_gradx, float *d_grady, int w, int h, int nc){
	// Get coordinates
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	// Get indices
	size_t idx = x + (size_t)w*y + (size_t)w*h*z;
	// Compute gradient
	if (x < w && y < h && z < nc){
		// Get neighbouring indices
		int x_high = x + 1;
		int y_high = y + 1;
		int x_low  = x - 1;
		int y_low  = y - 1;
		// Clamping
		if (x_high > w - 1){
			x_high = w - 1;
		}
		if (y_high > h - 1){
			y_high = h - 1;
		}
		if (x_low < 0){
			x_low = 0;	
		}
		if (y_low < 0){
			y_low = 0;
		}
		// Get indices of neighbouring indices
		size_t idx_x_high_y_high = x_high + (size_t)w*y_high + (size_t)w*h*z;
		size_t idx_x_high_y_low  = x_high + (size_t)w*y_low  + (size_t)w*h*z;
		size_t idx_x_low_y_high  = x_low  + (size_t)w*y_high + (size_t)w*h*z;
		size_t idx_x_low_y_low   = x_low  + (size_t)w*y_low  + (size_t)w*h*z;
		size_t idx_x_high_y_mid  = x_high + (size_t)w*y      + (size_t)w*h*z;
		size_t idx_x_low_y_mid   = x_low  + (size_t)w*y      + (size_t)w*h*z;
		size_t idx_x_mid_y_high  = x      + (size_t)w*y_high + (size_t)w*h*z;
		size_t idx_x_mid_y_low   = x      + (size_t)w*y_low  + (size_t)w*h*z;
		// Compute gradient	
		d_gradx[idx] = (3*d_imgIn[idx_x_high_y_high] + 10*d_imgIn[idx_x_high_y_mid] 
			  		  + 3*d_imgIn[idx_x_high_y_low]  - 3*d_imgIn[idx_x_low_y_high]  
					  - 10*d_imgIn[idx_x_low_y_mid]  - 3*d_imgIn[idx_x_low_y_low])/32;

		d_grady[idx] = (3*d_imgIn[idx_x_high_y_high] + 10*d_imgIn[idx_x_mid_y_high] 
					  + 3*d_imgIn[idx_x_low_y_high]  - 3*d_imgIn[idx_x_high_y_low]  
					  - 10*d_imgIn[idx_x_mid_y_low]  - 3*d_imgIn[idx_x_low_y_low])/32;
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
					d_imgOut[idx_3d] += d_kernel[idx_kernel_local] * d_imgIn[idx_kernel_global];
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
	float sigma = 0.5f;
	getParam("sigma", sigma, argc, argv);
	cout << "sigma: " << sigma << endl;
    // ### Define your own parameters here as needed    
	bool kernel_is_const = true;
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
	cv:: Mat mgradx(h, w, mIn.type());
	cv:: Mat mgrady(h, w, mIn.type());
	cv:: Mat mM1(h, w, CV_32FC1);
	cv:: Mat mM2(h, w, CV_32FC1);
	cv:: Mat mM3(h, w, CV_32FC1);
	cv:: Mat mT1(h, w, CV_32FC1);
	cv:: Mat mT2(h, w, CV_32FC1);
	cv:: Mat mT3(h, w, CV_32FC1);



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

	float *gradx = new float[nbytes];
	float *grady = new float[nbytes];
	float *m1 = new float[w*h];
	float *m2 = new float[w*h];
	float *m3 = new float[w*h];
	float *t1 = new float[w*h];
	float *t2 = new float[w*h];
	float *t3 = new float[w*h];

	float alpha = 0.0015f;
	float beta  = 0.001f;


	////////////////////////////////////////////////////////////////////// Block setting ///////////////////////////////////////////////////////////////////////

	dim3 block = dim3(128, 1, 1); 
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);


	////////////////////////////////////////////////////////////////////// Global Memory ////////////////////////////////////////////////////////////////////// 


	Timer timer; timer.start();

	// Arrays
	float *d_kernel;
	float *d_imgIn;
	float *d_imgOut;
	float *d_gradx;
	float *d_grady;
	float *d_m1;
	float *d_m2;
	float *d_m3;
	float *d_t1;
	float *d_t2;
	float *d_t3;
	float *d_eigen_value;
	// CUDA malloc
    cudaMalloc(&d_kernel, nbytes_kernel);			CUDA_CHECK;
    cudaMalloc(&d_imgIn, nbytes); 					CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytes); 					CUDA_CHECK;
	cudaMalloc(&d_gradx, nbytes);					CUDA_CHECK;
	cudaMalloc(&d_grady, nbytes);					CUDA_CHECK;
	cudaMalloc(&d_m1, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_m2, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_m3, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_t1, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_t2, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_t3, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_eigen_value, 2*sizeof(float));	CUDA_CHECK;
	// CUDA copy
    cudaMemcpy(d_kernel, kernel, nbytes_kernel, cudaMemcpyHostToDevice);	CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);			    CUDA_CHECK;
	// Convolution on original image
    convolution_global <<< grid, block >>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, w_kernel, h_kernel, kernel_is_const);	CUDA_CHECK;
	// Compute gradient of the convoluted image
	rotational_gradient <<< grid, block >>> (d_imgIn, d_gradx, d_grady, w, h, nc);											CUDA_CHECK;
	// Compute m1, m2, and m3
	compute_M <<< grid, block >>> (d_m1, d_m2, d_m3, d_gradx, d_grady, w, h, nc);											CUDA_CHECK;
	// Convolution on m1
	convolution_global <<< grid, block >>> (d_m1, d_t1, d_kernel, w, h, 1, w_kernel, h_kernel, kernel_is_const);			CUDA_CHECK;
	// Convolution on m2
	convolution_global <<< grid, block >>> (d_m2, d_t2, d_kernel, w, h, 1, w_kernel, h_kernel, kernel_is_const);			CUDA_CHECK;
	// Convolution on m3
	convolution_global <<< grid, block >>> (d_m3, d_t3, d_kernel, w, h, 1, w_kernel, h_kernel, kernel_is_const);			CUDA_CHECK;
	// Feature detection
	feature_detection <<< grid, block >>> (d_imgOut, d_eigen_value, d_t1, d_t2, d_t3, w, h, alpha, beta);					CUDA_CHECK;
	

	// Copy the results to host
    cudaMemcpy(imgOut, d_imgOut, nbytes, cudaMemcpyDeviceToHost); 		CUDA_CHECK;
	cudaMemcpy(gradx, d_gradx, nbytes, cudaMemcpyDeviceToHost);			CUDA_CHECK;
	cudaMemcpy(grady, d_grady, nbytes, cudaMemcpyDeviceToHost);			CUDA_CHECK;
	cudaMemcpy(m1, d_m1, w*h*sizeof(float), cudaMemcpyDeviceToHost);	CUDA_CHECK;		
	cudaMemcpy(m2, d_m2, w*h*sizeof(float), cudaMemcpyDeviceToHost);	CUDA_CHECK;
	cudaMemcpy(m3, d_m3, w*h*sizeof(float), cudaMemcpyDeviceToHost);	CUDA_CHECK;
	cudaMemcpy(t1, d_t1, w*h*sizeof(float), cudaMemcpyDeviceToHost); 	CUDA_CHECK;	
	cudaMemcpy(t2, d_t2, w*h*sizeof(float), cudaMemcpyDeviceToHost); 	CUDA_CHECK;	
	cudaMemcpy(t3, d_t3, w*h*sizeof(float), cudaMemcpyDeviceToHost); 	CUDA_CHECK;	

 	// Free memory
    cudaFree(d_imgIn);  		CUDA_CHECK;
    cudaFree(d_imgOut); 		CUDA_CHECK;
    cudaFree(d_kernel); 		CUDA_CHECK;
	cudaFree(d_gradx);  		CUDA_CHECK;
	cudaFree(d_grady);  		CUDA_CHECK;
	cudaFree(d_m1);				CUDA_CHECK;
	cudaFree(d_m2);				CUDA_CHECK;
	cudaFree(d_m3);				CUDA_CHECK;
	cudaFree(d_t1);				CUDA_CHECK;
	cudaFree(d_t2);				CUDA_CHECK;
	cudaFree(d_t3);				CUDA_CHECK;
	cudaFree(d_eigen_value);	CUDA_CHECK;


	// Type of processor
	processor = "GPU - global memory";
	cout << processor << endl;



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	timer.end();  float t = timer.get();
	cout << "time: " << t*1000 << " ms" << endl;


    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);


    // ### Display your own output images here as needed
	convert_layered_to_mat(mgradx, gradx);
	convert_layered_to_mat(mgrady, grady);
	convert_layered_to_mat(mM1, m1);
	convert_layered_to_mat(mM2, m2);
	convert_layered_to_mat(mM3, m3);
	convert_layered_to_mat(mT1, t1);
	convert_layered_to_mat(mT2, t2);
	convert_layered_to_mat(mT3, t3);

	/*
	showImage("grad_x", mgradx, 100+w+50, 150);
	showImage("grad_y", mgrady, 100+w+60, 150);
	showImage("m1", 10.f*mM1, 50, 200);
	showImage("m2", 10.f*mM2, 50 + w, 200);
	showImage("m3", 10.f*mM3, 50 + 2 * w, 200);
	showImage("t1", 10.f*mT1, 50, 250);
	showImage("t2", 10.f*mT2, 50 + w, 250);
	showImage("t3", 10.f*mT3, 50 + 2 * w, 250);
	*/


#ifdef CAMERA
    // end of camera loop
	}
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
	delete[] gradx;
	delete[] grady;
	delete[] m1;
	delete[] m2;
	delete[] m3;
	delete[] t1;
	delete[] t2;
	delete[] t3;
#endif

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
