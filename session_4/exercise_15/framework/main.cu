// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

// Exercise 15

// Written by: Jiho Yang (M.Sc student in Computational Science & Engineering)
// Matriculation number: 03675799

#include "helper.h"
#include <iostream>
#include <string>
#include <unistd.h>
using namespace std;


// uncomment to use the camera
//#define CAMERA

// Compute histogram in global memory
__global__ void compute_histogram_global(float *d_imgIn, int *d_histogram, int w, int h, int nc){
	// Get coordinates	
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	// Get indices
	int idx = x + (size_t)w*y + (size_t)w*h*z;
	int idx_hist = d_imgIn[idx] * 255.f;
	// Fill in histogram
	if (x < w && y < h && z < nc){
		atomicAdd(&d_histogram[idx_hist], 1);
	}
}

// Compute histogram in shared memory
__global__ void compute_histogram_shared(float *d_imgIn, int *d_histogram, int w, int h, int nc){
	// Create histogram in shared memory
	__shared__ int histogram_shared[256];
	// Get thex index of the thread
	int tid = threadIdx.x;
	// Initialise the histogram with zero
	if (tid < 256){
		histogram_shared[tid] = 0;
	}	
	__syncthreads();
	// Get coordinates
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	// Fill in histogram
	if (x < w && y < h && z < nc){
		int idx = x + (size_t)w*y + (size_t)w*h*z;
		int idx_hist = d_imgIn[idx] * 255.f;
		atomicAdd(&histogram_shared[idx_hist], 1);
	}
	__syncthreads();
	// Update the histogram in global memory
	if (tid < 256){
		atomicAdd(&d_histogram[tid], histogram_shared[tid]);
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
    // ### Define your own parameters here as needed    

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

	int nbytes = (size_t)w*h*nc*sizeof(float);

    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

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

	// Histogram setup
	int histSize = 256;
	// Allocate memory on host
	int *histogram = new int[histSize];
	// Nbytes
	int nbytes_hist = histSize * sizeof(int);


	// Processor type
	string processor;


	////////////////////////////////////////////////////////////////////// Block setting ///////////////////////////////////////////////////////////////////////

	dim3 block = dim3(256, 1, 1); 
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);


	// Device	
	Timer timer; timer.start();

	// Arrays
	float *d_imgIn;
	float *d_imgOut;
	int *d_histogram;
	// CUDA
    cudaMalloc(&d_imgIn, nbytes); 			CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytes); 			CUDA_CHECK;
	cudaMalloc(&d_histogram, nbytes_hist);	CUDA_CHECK;
	// Initialise histogram
	for (size_t i = 0; i < histSize; i++){
		histogram[i] = 0;
	}
	// Copy to device
    cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);			    	CUDA_CHECK;
	cudaMemcpy(d_histogram, histogram, nbytes_hist, cudaMemcpyHostToDevice);	CUDA_CHECK;
	// Histogram on shared memory
	compute_histogram_shared <<< grid, block >>> (d_imgIn, d_histogram, w, h, nc);	CUDA_CHECK;
	processor = "GPU - Shared";
	cout << processor << endl;
	// Histogram on global memory
	//compute_histogram_global <<< grid, block >>> (d_imgIn, d_histogram, w, h, nc);	CUDA_CHECK;
	//processor = "GPU - Global";
	//cout << processor << endl;
	// Copy the results to hist
	cudaMemcpy(histogram, d_histogram, nbytes_hist, cudaMemcpyDeviceToHost);		CUDA_CHECK;
	// Measure end time
	timer.end();  float t = timer.get();
	cout << "time: " << t*1000 << " ms" << endl;
	// Visualise
	convert_layered_to_mat(mIn, imgIn);
	showImage("Input", mIn, 100, 100);
	showHistogram256("Histogram", histogram, 1000, 100);
 	// Free memory
    cudaFree(d_imgIn);  		CUDA_CHECK;
    cudaFree(d_imgOut); 		CUDA_CHECK;
	cudaFree(d_histogram);		CUDA_CHECK;

#ifdef CAMERA
    // end of camera loop
	}
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

	// free allocated arrays
#ifdef CAMERA
	delete[] imgIn;
	delete[] imgOut;
#else
	delete[] imgIn;
	delete[] imgOut;
	delete[] histogram;
#endif

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
