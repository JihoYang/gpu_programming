<<Exercise 10>>

how to compile : do 'make' in build_cmake

how to run it: copy and paste the follinwg command

./main -i ~/cuda_ss17/images/dog.png

you can choose any images, but dog seems to give better contrast/effects of blurring, etc

--Solutions--

6. If time step is taken to be high (above stability criterion) the explicit euler iteration diverges, hence giving completely wrong results. If a very large N is chosen, I assume the output image will reach towards NaN.
	Anyhow if tau is stable, very large N would mean reaching towards steady state. I tested with N = 60000 and the output showed four regions of different colours but I am guessing as N -> infinity the output image will contain a single colour (the colour which is the most distributed in the original image);

7. Gaussian convolution gives similar effects (same in my eyes) as the diffusion problem.

8. Huber diffusion gives a more pastel drawing like texture to the processed image, preserving the edges more (As if one was to draw the same image but with pastel/paint);
	The next diffusion operator (exp(-s^2/eps)/eps) is like huber diffusion but with stronger edge preserving effects.


To try different diffusion operators, comment/uncomment the operators on line 88 - 96
The number of iterations (N), time step (tau), and sigma for convolution are to be modified on line 284 - 287
