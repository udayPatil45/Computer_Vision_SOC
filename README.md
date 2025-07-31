# Computer_Vision_SOC

WEEK 1

In first week i revised my python basics in which it includes if-else statements , loops concepts ( while , for) and datastructures like list , sets , strings , dictionary etc. and also related to file handling in which write , read , append modes. 

Then i learned about NumPy Library :
in which i learned about NumPy Arrays in which :  
Creating arrays using  (np.array, np.zeros, np.ones, np.arange, np.linspace)  , Then  
Array types: 1D, 2D, and multidimensional arrays ,Array attributes (shape, ndim, size, itemsize) ,Array Indexing and Slicing ,Boolean indexing and masking ,Array Operations ,Element-wise operations ,Broadcasting, Vectorized operations  ,Comparison and logical operations 
after this array operation i learned about the mathematical structures of numPy Library in which:   
Mathematical and Statistical Functions     
Aggregate functions (sum, mean, std, var, min, max) , Axis-based operations  ,Mathematical functions (np.exp, np.sqrt, np.log, np.sin, np.cos, etc.)  ,Reshaping and Manipulating Arrays  ,Linear Algebra with NumPy ,Matrix operations: dot product, matrix multiplication , Solving linear equations , Sampling and probability distributions
Difference between shallow and deep copy    
Then i read about some Random Functions :    
Random Module in NumPy   
np.random.rand, randn, randint    
Seed setting (np.random.seed())    
    

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


WEEK 4 

in this week i learned further more about image processing. this week was mainly code based so uploaded that work in image processing.ipynb file.
to see 

LINK - [ https://colab.research.google.com/github/udayPatil45/Computer_Vision_SOC/blob/main/image_processing.ipynb ]
same file i also uploaded in repo use this link if that block is not opening.


 Week 5: Foundations of Neural Networks and Backpropagation


🔹 Focus
This week was dedicated to developing a strong conceptual and mathematical understanding of how neural networks operate and learn using gradient descent and backpropagation. Unlike Weeks 1–4, this content shifts toward the core foundations of machine learning and deep learning, which are essential for modern computer vision and AI applications.

🔹 Key Learnings
1. Structure of Neural Networks
A neural network is built from layers of neurons:
Input Layer: Receives the raw data (e.g., pixels of an image).
Hidden Layers: Perform transformations using weights and biases.
Output Layer: Produces the prediction (e.g., class probabilities).

2. Gradient Descent
The goal: Minimize the loss function (difference between predicted and true values).
Gradient descent updates weights in the opposite direction of the gradient

3. Backpropagation Intuition
Forward Pass: Compute outputs from inputs.
Backward Pass: Use chain rule of calculus to compute gradients of loss w.r.t. each parameter.
Update Parameters: Apply gradient descent to improve the network’s performance.
Backprop ensures efficient computation of derivatives in deep networks.

4. Connection to Applications
These principles are the foundation of image classification tasks, where models learn to map pixel inputs → class probabilities.
Advanced architectures (CNNs, ResNets, Transformers) are built on these basics.


Week 6 - Neural Networks and Image Classification
Overview:
In Week 6, I explored how neural networks are applied to image classification tasks using PyTorch, with supporting materials also available in TensorFlow. This week focused on understanding how neural networks detect, classify, and interpret visual patterns.

Key Learnings:

Image Classification Using Simple Neural Networks:
Understood how to build basic image classifiers from scratch using PyTorch.
Learned how neural networks process image data and classify them into categories.

Deep Learning Visual Pattern Perception:
Studied how deep models detect edges, textures, and complex features.
Visualized internal layer activations to better grasp feature extraction.

Using Custom Datasets:
Learned the process of preparing and loading custom datasets.
Gained hands-on experience with Dataset and DataLoader in PyTorch.

Foundational Computer Vision Techniques:
Applied essential vision techniques like transformations, normalization, and augmentation.
Implemented modular and scalable deep learning pipelines.


Final Model


In this focused on replicating a cutting-edge machine learning research paper—specifically, the Vision Transformer (ViT) model. This experience helped me bridge the gap between theory and practice by implementing a state-of-the-art deep learning architecture from scratch using PyTorch.

1. Understanding Paper Replication
I began by learning what it means to replicate a machine learning paper. Instead of just reading theoretical concepts, I practiced interpreting the original research—math equations, architecture diagrams, and textual descriptions—and converting them into functioning, testable code. This process sharpened my ability to read and understand complex academic literature and made me more confident in applying research-level techniques to real-world problems.

2. Deep Dive into Vision Transformers (ViT)
The core of the project was understanding and implementing the Vision Transformer (ViT) architecture. Unlike traditional Convolutional Neural Networks (CNNs), ViT splits an image into smaller fixed-size patches and treats each patch like a token in NLP. I learned how these patches are linearly projected into embedding vectors and how positional embeddings are added to retain spatial information. The embedded patches are then passed through multiple transformer encoder blocks, each containing multi-head self-attention layers and feedforward networks.

This approach fundamentally changed how I view image classification, showing that CNNs are not the only way to process visual data. I also understood how attention mechanisms help models focus on the most relevant parts of an image when making predictions.

3. PyTorch Implementation Skills
Throughout the project, I significantly improved my PyTorch skills. I implemented custom modules like:

Patch Embedding layers using Conv2D

Transformer encoder blocks using LayerNorm, MultiheadAttention, and MLPs

The complete ViT model, including its classification head

I also gained experience in organizing my model code into reusable, modular components—a best practice in real-world machine learning development.

4. Dataset Handling and Training
The model was trained on a small but realistic dataset called FoodVision Mini, containing images of pizza, steak, and sushi. I learned to preprocess images using torchvision.transforms, load them efficiently using DataLoader, and train the model using a standard PyTorch loop with CrossEntropyLoss and the Adam optimizer. I monitored training and validation accuracy to assess performance and adjusted hyperparameters where necessary.

5. Broader Takeaways
This project deepened my understanding of:

How to approach paper replication projects methodically
The structure and purpose of modern computer vision research
The real-world application of transformer-based models
Writing clean, modular, and reusable deep learning code


Conclusion
By completing this Final model, I developed a strong foundation in paper replication, model implementation, and transformer-based vision models. I now feel more confident in reading academic papers, building models from scratch, and experimenting with innovative architectures for various machine learning tasks.

