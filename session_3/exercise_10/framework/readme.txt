<<Exercise 10>>

how to compile : do 'make' in build_cmake

how to run it: copy and paste the follinwg command

./main -i <<directory of image>>

To change the tau and N refer to line 472
For alpha and C refer to line 598


--Solutions--

3. Tried different choices of C, alpha, and N and couldn't get a significant difference. This was due to very small diffusion tensor values that returns a very small value of gradients (and hence divergence). Often the divergences were zero, hence not updating the image (and if not zero, very close to zero).

I tested the code with G being an identity matrix and could observe a laplacian diffusion, and therefore the other parts of the codes are correct. I also checked my eigenvalues and eigenvectors (for structure tensors of couple of pixels) by comparing the results with Matlab's eig(A), and they matched. The only difference was in eigenvector, where different scaling factor was applied (the ratio of the elements, however, was the same, hence same basis). 

I tried to scale the eigenvectors to increase the values of diffusion tensor, based on the assumption that eigenvalues can be scaled without loosing the properties, and still no change in diffusion was observed (and if some bad tau values are used, the solution diverged). 

I tried to reduce the C value so as to increase my mu_2, but in the scope of float precision, this didn't work out (if I decreased even more it becomes zero).

I tried to change all the functions and values into double, and could observe some minor diffusion, but again, not significant). 

Overall, I have a impression that anisotropic diffusion is very slow and weak.

** I append the results using the following setup

alpha = 0.9
C = 0.0000005
tau = 0.001
N = 20000

It's hard to distinguish the diffusion just by having those two images next to each other. Try opening on image and switch to the other one by entering space, and the diffusion is more visible.


