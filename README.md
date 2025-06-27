# Computer_Vision_SOC

WEEK 1

In first week i revised my python basics in which it includes if-else statements , loops concepts ( while , for) and datastructures like list , sets , strings , dictionary etc. and also related to file handling in which write , read , append modes. 

Then i learned about NumPy Library :
NumPy Arrays ,  
Creating arrays (np.array, np.zeros, np.ones, np.arange, np.linspace)  ,  
Array types: 1D, 2D, and multidimensional arrays ,Data types (dtype)  ,Array attributes (shape, ndim, size, itemsize) ,Array Indexing and Slicing ,Indexing 1D, 2D arrays ,Boolean indexing and masking ,Array Operations ,Element-wise operations ,Broadcasting, Vectorized operations  ,Comparison and logical operations    
Mathematical and Statistical Functions     
Aggregate functions (sum, mean, std, var, min, max)    
Axis-based operations     
Mathematical functions (np.exp, np.sqrt, np.log, np.sin, np.cos, etc.)   
Reshaping and Manipulating Arrays     
Difference between shallow and deep copy    
arr.copy() vs simple assignment    
Random Module in NumPy   
np.random.rand, randn, randint    
Seed setting (np.random.seed())    
Sampling and probability distributions    
Linear Algebra with NumPy    
Matrix operations: dot product, matrix multiplication
Solving linear equations   
File I/O    
Saving and loading arrays   

WEEK 2-3

In next weeks i learned about signal processing and basics of image processing from sources like GFG and from video lectures provided.
I learned
1. Fourier Transform and Frequency Domain Analysis
I understood that Fourier Transform (FT) is a mathematical tool that transforms a signal/image from the spatial domain (pixel values) to the frequency domain, where patterns, edges, and textures can be analyzed more efficiently.
In the frequency domain, low-frequency components represent smooth variations (like the background), and high-frequency components represent abrupt changes (like edges and noise).
The 2D Discrete Fourier Transform (DFT) is used for images
 I used NumPy (np.fft.fft2) to perform FFT, and np.fft.fftshift to center the zero-frequency component.
I also learned about zero-padding: padding an image with zeros to increase its size, which improves frequency resolution and helps avoid wraparound errors in circular convolution.

2. Convolution Operations and Their Applications
I studied the operation of convolution, which is a fundamental building block in both classical image processing and deep learning.
Convolution involves sliding a filter (also called a kernel or mask) over the image to produce effects like blurring, sharpening, edge detection, etc.
The mathematical operation of convolution:
I explored correlation (similar to convolution but without flipping the kernel), and how libraries like OpenCV use it.
Applications of convolution I explored:
Blurring using a box or Gaussian filter
Edge detection using Sobel, Prewitt, or Laplacian filters
Noise reduction using averaging or Gaussian smoothing

3. Filters: Types and Implementation
I studied different spatial and frequency domain filters, and implemented some of them in code:
Low-Pass Filters
Allow low-frequency components to pass; smooth the image.
Examples: Gaussian, Ideal, Butterworth

Used for blurring, denoising
 High-Pass Filters
Allow high-frequency components; highlight edges and sharp transitions.
Examples: High-pass Gaussian, Ideal high-pass, Laplacian

Used for edge detection, sharpening
Band-Pass and Band-Stop Filters
Allow or suppress frequencies in a specific range.

I implemented these filters in the frequency domain using multiplication with the Fourier-transformed image, and then used inverse FFT to return to the spatial domain.
