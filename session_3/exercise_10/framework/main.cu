// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

// Exercise 10

// Written by: Jiho Yang (M.Sc student in Computational Science & Engineering)
// Matriculation number: 03675799

#include "helper.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <unistd.h>
using namespace std;

const float pi = 3.141592653589793238462f;

// uncomment to use the camera
//#define CAMERA

// Compute gradient
__global__ void compute_gradient(float *d_gradx, float *d_grady, float *d_imgIn, int w, int h, int nc){
	// Get x y z pixel coordinates in 3D kernel
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int z = threadIdx.z + blockIdx.z*blockDim.z;
	// Get high indices
	size_t x_high = x + 1 + (size_t)w*y + (size_t)h*w*z;
	size_t y_high = x + (size_t)w*(y+1) + (size_t)h*w*z;
	size_t idx = x + (size_t)w*y + (size_t)h*w*z;
	// Ensure no threads are out of problem domain
	if (x < w && y < h && z < nc){
	// Compute gradient
		if (x < w-1){
			d_gradx[idx] = d_imgIn[x_high] - d_imgIn[idx];
		} else
			d_gradx[idx] = 0;
		if (y < h-1){
			d_grady[idx] = d_imgIn[y_high] - d_imgIn[idx];
		} else
			d_grady[idx] = 0;
	}
}

// Compute L2 norm
__device__ void compute_norm(float *d_norm, float *d_vec1, float *d_vec2, int w, int h, int nc){
	// Temporary variable for norm
	float sqrd1 = 0;
	float sqrd2 = 0;
	float val1, val2;
	// Get coordinates
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	// Get index
	int idx = x + (size_t)w*y;
	// Compute norm
	if (x < w && y < h){
		for (size_t c = 0; c < nc; c++){
			// Get index
			size_t idx_3d = idx + (size_t)w*h*c;
			// Compute L2 norm
			val1 = d_vec1[idx_3d];
			val2 = d_vec2[idx_3d];
			sqrd1 += val1*val1;
			sqrd2 += val2*val2;
		}
		d_norm[idx] = sqrtf(sqrd1*sqrd1 + sqrd2*sqrd2);
	}
}

// Compute divergence
__global__ void compute_divergence(float *d_div, float *d_gradx, float *d_grady, int w, int h, int nc){
	// Get x y z pixel coordinates in 3D kernel
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int z = threadIdx.z + blockIdx.z*blockDim.z;
	// Get low indices
	size_t idx = x + (size_t)w*y + (size_t)h*w*z;
	size_t x_low = x-1 + (size_t)w*y + (size_t)h*w*z;
	size_t y_low = x + (size_t)w*(y-1) + (size_t)h*w*z;
	// Temporary values 
	float v_x, v_y;
	// Ensure no threads are out of problem domain
	if (x < w && y < h && z < nc){
		// Compute divergence
		if (x > 1){
			v_x = d_gradx[idx] - d_gradx[x_low];
			
		} else
			v_x = 0;
		if (y > 1){
			v_y = d_grady[idx] - d_grady[y_low];
		} else
			v_y = 0;
		// Sum gradients
		d_div[idx] = v_x + v_y;
	}

	if (idx == 100){
		printf("Divergence = %f\n", d_div[idx]);
	}
}
	
// Convolution on global memory
__global__ void convolution_global(float *d_imgIn, float *d_imgOut, float *d_kernel, int w, int h, int nc, int w_kernel, int h_kernel){
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

// Compute eigenvalue of a 2 by 2 matrix
__device__ void compute_eigenvalue(float &eigen_value_0, float &eigen_value_1, float &eigen_vector_0, float &eigen_vector_1, float &eigen_vector_2, float &eigen_vector_3, float d_t1_val, float d_t2_val, float d_t3_val, int w, int h){
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
	eigen_value_0 = T/2 + sqrtf(T*T/4-D);
	eigen_value_1 = T/2 - sqrtf(T*T/4-D);
	// Sort eigenvalue array (val_1 > val_2)
	if (eigen_value_0 < eigen_value_1){
		float swap = eigen_value_0;
		eigen_value_0 = eigen_value_1;
		eigen_value_1 = swap;
	}
	// Compute eigenvectors
	/*
	if (c != 0){
		eigen_vector_0 = eigen_value_0 - d;
		eigen_vector_1 = c;
		eigen_vector_2 = eigen_value_1 - d;
		eigen_vector_3 = c;
	}
	*/
	else if (b != 0){
		eigen_vector_0 = b;
		eigen_vector_1 = eigen_value_0 - a;
		eigen_vector_2 = b;
		eigen_vector_3 = eigen_value_1 - a;
	}
	else if (b == 0 && c == 0){
		eigen_vector_0 = 1;
		eigen_vector_1 = 0;
		eigen_vector_2 = 0;
		eigen_vector_3 = 1;
	}

	// Scale eigenvector
	eigen_vector_0 = 1*eigen_vector_0;	
	eigen_vector_1 = 1*eigen_vector_1;	
	eigen_vector_2 = 1*eigen_vector_2;	
	eigen_vector_3 = 1*eigen_vector_3;	

	// Get coordinates
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	// Get index
	size_t idx = x + (size_t)w*y + (size_t)w*h*z;
/*
	if (idx == 0){
		printf("a = %f\n", a);
		printf("b = %f\n", b);
		printf("c = %f\n", c);
		printf("d = %f\n", d);
	}
*/
}

// Apply anisotropic diffusion
__global__ void apply_diffusion(float *d_gradx, float *d_grady, float *d_imgIn, float alpha, float C, float *d_t1, float *d_t2, float *d_t3, int w, int h, int nc){
	// Define eigenvalue and eigenvector
	float eigen_value_0, eigen_value_1;
	float eigen_vector_0, eigen_vector_1, eigen_vector_2, eigen_vector_3;
	// Get coordinates
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	// Get index
	size_t idx = x + (size_t)w*y + (size_t)w*h*z;
	size_t idx_2d = x + (size_t)w*y;
	// Compute eigenvalues and eigenvector
	if (x < w && y < h && z < nc){
		compute_eigenvalue(eigen_value_0, eigen_value_1, eigen_vector_0, eigen_vector_1, eigen_vector_2, eigen_vector_3, d_t1[idx_2d], d_t2[idx_2d], d_t3[idx_2d], w, h);
	}
	__syncthreads();
	// Get Mu
	float mu_1 = alpha;
	float mu_2;
	if (eigen_value_0 == eigen_value_1){
		mu_2 = alpha;
	} else{
		float eigdif = eigen_value_0 - eigen_value_1;
		float inside = -C/(eigdif*eigdif);
		mu_2 =  alpha + (1 - alpha)*exp(inside);
	}
	// Get diffusion tensor
	float G[4];
	G[0] = mu_1*eigen_vector_0*eigen_vector_0 + mu_2*eigen_vector_2*eigen_vector_2;
	G[1] = mu_1*eigen_vector_0*eigen_vector_1 + mu_2*eigen_vector_2*eigen_vector_3;
	G[2] = mu_1*eigen_vector_1*eigen_vector_0 + mu_2*eigen_vector_3*eigen_vector_2;
	G[3] = mu_1*eigen_vector_1*eigen_vector_1 + mu_2*eigen_vector_3*eigen_vector_3;

	G[0] = 1;
	G[1] = 0;
	G[2] = 1;
	G[3] = 0;

	__syncthreads();

/*
	if (idx == 0){
		printf("Before diffusion\n");
		printf("d_gradx = %f\n", d_gradx[idx]);
		printf("d_grady = %f\n", d_grady[idx]);
	}
*/

	// Update gradient
	if (x < w && y < h && z < nc){
		d_gradx[idx] = G[0]*d_gradx[idx] + G[1]*d_grady[idx];
		d_grady[idx] = G[2]*d_gradx[idx] + G[3]*d_grady[idx];
	}
	
	if (idx == 100){
		printf("After diffusion\n");
		printf("d_gradx = %f\n", d_gradx[idx]);
		printf("d_grady = %f\n", d_grady[idx]);


/*
		printf("G[0] = %f\n", G[0]);
		printf("G[1] = %f\n", G[1]);
		printf("G[2] = %f\n", G[2]);
		printf("G[3] = %f\n", G[3]);

		printf("eigenvalue_0 = %f\n", eigen_value_0);
		printf("eigenvalue_1 = %f\n", eigen_value_1);
		printf("eigenvector_0 = %f\n", eigen_vector_0);
		printf("eigenvector_1 = %f\n", eigen_vector_1);
		printf("eigenvector_2 = %f\n", eigen_vector_2);
		printf("eigenvector_3 = %f\n", eigen_vector_3);

		printf("Mu_1 = %f\n", mu_1);
		printf("Mu_2 = %f\n", mu_2);
*/
	}
}

// Update image
__global__ void update_image(float *d_imgIn, float *d_div, float tau, int w, int h, int nc){
	// Get coordinates	
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	// Get index
	size_t idx = x + (size_t)w*y + (size_t)w*h*z;

	if (idx == 100){
		printf("Before update\n");
		printf("d_imgIn = %f\n", d_imgIn[idx]);
	}


	if (x < w && y < h && z < nc){	
		// Update image	
		d_imgIn[idx] += tau * d_div[idx];
	}

	if (idx == 100){
		printf("After update\n");
		printf("d_imgIn = %f\n", d_imgIn[idx]);
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
	if (x < w && y < h){
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



	// Diffusion
	float tau = 0.02f;
	int	N = 500;
	// Convolution kernel
	float sigma = 0.5f;
	float phi = 3.0f;
	getParam("sigma", sigma, argc, argv);
	cout << "sigma: " << sigma << endl;
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
	// Define kernel dimensions
	int r = ceil(3*sigma);
	int w_kernel = r * 2 + 1;	  //windowing
	int h_kernel = w_kernel;  	  //Square kernel
	int r_phi = ceil(3*phi);
	int w_kernel_phi = r_phi * 2 + 1;
	int h_kernel_phi = w_kernel_phi;
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
	cv:: Mat mdiv(h, w, mIn.type());

	cv:: Mat mt1(h, w, CV_32FC1);
	cv:: Mat mt2(h, w, CV_32FC1);
	cv:: Mat mt3(h, w, CV_32FC1);



    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

	// Get array memory
	int nbytes = w * h * nc * sizeof(float);
	int nbytes_kernel = w_kernel * h_kernel * sizeof(float);
	int nbytes_kernel_phi = w_kernel_phi * h_kernel_phi * sizeof(float);
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

		
	////////////////////////////////////////////////////////////////////// Block setting ///////////////////////////////////////////////////////////////////////

	dim3 block = dim3(128, 1, 1); 
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1));


	float alpha = 0.01f;
	float C = 0.0000005f;

	Timer timer; timer.start();

	////////////////////////////////////////////////////////////////////// Arrays (Device) /////////////////////////////////////////////////////////////////////

	// Kernel
	float *d_kernel;
	float *d_kernel_phi;
	// Images
	float *d_imgIn;
	float *d_imgOut;
	// Gradients
	float *d_gradx;
	float *d_grady;
	// Gradients for structure tensor
	float *d_gradx_tensor;	
	float *d_grady_tensor;
	float *d_m1;
	float *d_m2;
	float *d_m3;
	float *d_t1;
	float *d_t2;
	float *d_t3;
	// Norm
	float *d_norm;
	// Divergence
	float *d_div;

	////////////////////////////////////////////////////////////////////// Arrays (Host) /////////////////////////////////////////////////////////////////////

	// Kernel
	float *kernel = new float[nbytes_kernel]; 
	float *kernel_phi = new float[nbytes_kernel_phi];
	// Structure tensor
	float *m1 	 = new float[w*h];
	float *m2 	 = new float[w*h];
	float *m3 	 = new float[w*h];
	float *t1 	 = new float[w*h];
	float *t2 	 = new float[w*h];
	float *t3 	 = new float[w*h];
	// Gradient
	float *gradx = new float[nbytes];
	float *grady = new float[nbytes];
	// Divergence
	float *div   = new float[nbytes];

	////////Create kernel
	get_kernel(kernel,  w_kernel, h_kernel, pi, sigma);
	get_kernel(kernel_phi, w_kernel_phi, h_kernel_phi, pi, phi);
	// Processor type
	string processor;


	////////////////////////////////////////////////////////////////////////// CUDA	////////////////////////////////////////////////////////////////////////// 
	
	// CUDA malloc
	// Kernel
    cudaMalloc(&d_kernel, nbytes_kernel);			CUDA_CHECK;
	cudaMalloc(&d_kernel_phi, nbytes_kernel_phi);	CUDA_CHECK;
	// Images
    cudaMalloc(&d_imgIn, nbytes); 					CUDA_CHECK;
    cudaMalloc(&d_imgOut, nbytes); 					CUDA_CHECK;
	// Gradients
	cudaMalloc(&d_gradx, nbytes);					CUDA_CHECK;
	cudaMalloc(&d_grady, nbytes);					CUDA_CHECK;
	// Gradients for structure tensor
	cudaMalloc(&d_gradx_tensor, nbytes);			CUDA_CHECK;
	cudaMalloc(&d_grady_tensor, nbytes);			CUDA_CHECK;
	cudaMalloc(&d_m1, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_m2, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_m3, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_t1, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_t2, w*h*sizeof(float));			CUDA_CHECK;
	cudaMalloc(&d_t3, w*h*sizeof(float));			CUDA_CHECK;
	// Norm
	cudaMalloc(&d_norm, w*h*sizeof(float));			CUDA_CHECK;
	// Divergence
	cudaMalloc(&d_div,   nbytes);					CUDA_CHECK;

	// CUDA copy
    cudaMemcpy(d_kernel, kernel, nbytes_kernel, cudaMemcpyHostToDevice);												CUDA_CHECK;
	cudaMemcpy(d_kernel_phi, kernel_phi, nbytes_kernel_phi, cudaMemcpyHostToDevice);									CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);			    											CUDA_CHECK;

	// Iterations
	for (size_t i = 0; i < N; i++){
		// Initial convolution - structure tensor
		convolution_global <<< grid, block >>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, w_kernel, h_kernel);					CUDA_CHECK;
		// Compute gradient of the convoluted image	- structure tensor
		rotational_gradient <<< grid, block >>> (d_imgIn, d_gradx_tensor, d_grady_tensor, w, h, nc);						CUDA_CHECK;
		// Compute m1, m2, and m3 - structure tensor
		compute_M <<< grid, block >>> (d_m1, d_m2, d_m3, d_gradx_tensor, d_grady_tensor, w, h, nc);							CUDA_CHECK;
		// Convolution on m1 - structure tensor
		convolution_global <<< grid, block >>> (d_m1, d_t1, d_kernel_phi, w, h, 1, w_kernel_phi, h_kernel_phi);						CUDA_CHECK;
		// Convolution on m2 - structure tensor
		convolution_global <<< grid, block >>> (d_m2, d_t2, d_kernel_phi, w, h, 1, w_kernel_phi, h_kernel_phi);						CUDA_CHECK;
		// Convolution on m3 - structure tensor
		convolution_global <<< grid, block >>> (d_m3, d_t3, d_kernel_phi, w, h, 1, w_kernel_phi, h_kernel_phi);						CUDA_CHECK;
		// Compute gradient
		compute_gradient <<< grid, block >>> (d_gradx, d_grady, d_imgIn, w, h, nc);													CUDA_CHECK;
		// Apply diffusion tensor
		apply_diffusion <<< grid, block >>> (d_gradx, d_grady, d_imgIn, alpha, C, d_t1, d_t2, d_t3, w, h, nc);				CUDA_CHECK;
		// Compute divergence
		compute_divergence <<< grid, block >>> (d_div, d_gradx, d_grady, w, h, nc);											CUDA_CHECK;
		// Update image
		update_image <<< grid, block >>> (d_imgIn, d_div, tau, w, h, nc);													CUDA_CHECK;
	}

	// Copy the results to host
	cudaMemcpy(imgOut, d_imgIn, nbytes, cudaMemcpyDeviceToHost);		CUDA_CHECK;
	cudaMemcpy(gradx,  d_gradx_tensor, nbytes, cudaMemcpyDeviceToHost);		CUDA_CHECK;
	cudaMemcpy(grady,  d_grady_tensor, nbytes, cudaMemcpyDeviceToHost);		CUDA_CHECK;
	cudaMemcpy(div,    d_div,   nbytes, cudaMemcpyDeviceToHost);		CUDA_CHECK;

	cudaMemcpy(t1, d_t1, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(t2, d_t2, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(t3, d_t3, w*h*sizeof(float), cudaMemcpyDeviceToHost);



 	// Free memory
    cudaFree(d_imgIn);  		CUDA_CHECK;
    cudaFree(d_imgOut); 		CUDA_CHECK;
    cudaFree(d_kernel); 		CUDA_CHECK;
	cudaFree(d_div);			CUDA_CHECK;
	cudaFree(d_gradx);  		CUDA_CHECK;
	cudaFree(d_grady);  		CUDA_CHECK;
	cudaFree(d_norm);			CUDA_CHECK;

	cudaFree(d_t1);
	cudaFree(d_t2);
	cudaFree(d_t3);


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
	showImage("Diffusion", mOut, 100+w+40, 100);


    // ### Display your own output images here as needed

	convert_layered_to_mat(mgradx, gradx);
	convert_layered_to_mat(mgrady, grady);
	convert_layered_to_mat(mdiv, div);

	convert_layered_to_mat(mt1, t1);
	convert_layered_to_mat(mt2, t2);
	convert_layered_to_mat(mt3, t3);

	showImage("t1", 10.f*mt1, 50, 250);
	showImage("t2", 10.f*mt2, 50 + w, 250);
	showImage("t3", 10.f*mt3, 50 + 2 * w, 250);

	//showImage("grad_x", mgradx, 100+w+50, 150);
	//showImage("grad_y", mgrady, 100+w+60, 150);
	//showImage("div", mdiv, 100+w+80, 200);


/*
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

#endif

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
