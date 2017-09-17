<<Exercise 3>>

How to execute:

./main -i <directory of image>

To change the settings for CPU or GPU version, refer to 160 - 190
 
     3. Average time for CPU (5 repeats) = 39.9294 ms
        Average time for GPU (5 repeats whole execution ) = 1.9806 ms
        Average time for GPU (5 repeats just kernel execution) = 0.1725 ms
        Allocating and copying memory takes significant amount of time for GPU operatio    ns while the actual computation does not take much.

	4. GPU time with blockDim.x = 32 is 1.223 ms
	   GPU time with blockDim.x = 64 is 0.663 ms
	   GPU time with blockDim.x = 96 is 0.378 ms
	   GPU time with blockDim.x = 128 is 0.383 ms
	   GPU time with blockDim.x = 256 is 0.244 ms

		BlockDim.x with 96 gave the minimum execution time

