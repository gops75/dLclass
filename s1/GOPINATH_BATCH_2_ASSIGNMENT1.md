# Session 1 - Assignment 1
*Gopinath Venkatesan -- Batch 2*

## Objective:
To discuss few terms used in Convolutional Neural Network (CNN) and summarize two of them for easy interpretations.

### Keyterms used in CNN:
Few key terms used extensively in the CNN are listed below:
1. Convolution
2. Filters / Kernels
3. Epochs
4. 1 x 1 Convolution
5. 3 x 3 Convolution
6. Feature Maps
7. Feature Engineering
8. Activation function
9. Receptive field

#### 1. Convolution
Convolution is used in engineering mathematics to denote the sum of infinitesimally small discrete areas produced by the product of two functions $f$ and $g$, over a period of time $t$, denoted by $(f*g)(t)$, and is expressed as below:

$$(f*g)(t) = \int^{+\infty}_{-\infty} f(\tau)g(t-\tau)d\tau$$

From the above equation, we see that the second function $g$ is first reversed in time (denoted by *$-\tau$*), and shifted by time equivalent of *t* and then is multiplied with $f(\tau)$ at each instant of $\tau$, and summed to get the result.

In practical applications, it is thought of smoothing out the abrupt discontinuity at the ends of a function $f$. Say we have a square or rectangular pulse $f$ as shown below in the Figure 1 [1], we can use $g$ (which is also square in this case) to convolve $f$ to an equivalent triangular pulse, smoothing out the end variations in $f$.

![Convolution - Example](https://upload.wikimedia.org/wikipedia/commons/6/6a/Convolution_of_box_signal_with_itself2.gif "f convolved by g, (f*g)(t)")

The term 'Convolution', as applied to the computer vision applications, also has the same nature but since each pixel is a discrete value, we may find that the integral sign is missing, and instead will be seeing the summation carried over the dot product of two matrices. And here, depending upon the nature of convolute $g$, the output nature also varies. Suppose the convolute matrix $g$ is an edge detection filter matrix, then the edges of the images are filtered. A Convolutional Neural Network (CNN) may use several such detection filters for identifying and filtering specific attributes of an image for classification purposes.

#### 2. Filter / Kernel
A **filter** or a **kernel** is nothing but a weight matrix with weights so arranged to give higher weights to the interested attributes of the image. In CNNs, every neural network layer is some sort of detection filter or kernel to identify and retrieve specific patterns/features of interest. 

Usually the input image matrix is $f$ which is a $n$ x $n$ pixels, and the second matrix $g$ is the filter matrix, often a 3 x 3 matrix, characterizes the nature of operation to be done on the input matrix $f$. Per the convolution equation, we take the filter matrix and traverse through the input matrix selecting the 3 x 3 subset recursively from the input matrix and finding the summed dot product of this subset matrix with the filter matrix, to yield a resultant convolute matrix. Suppose we start with $n$ x $n$ input matrix, operated by the 3 x 3 filter matrix, we get a resultant matrix of ($n-2$) x ($n-2$).

Successive to each convolution operation in the form of detection filters, it is customary to apply pooling (maxPooling) operation to reduce the size of the image matrix. The maxPooling operation takes the maximum from the 3 x 3 convolute matrix, thus only one number is retained for every 3 x 3 convolute matrix, yielding a reduced size for the output matrix. If we apply maxPooling on the ($n-2$) x ($n-2$) image matrix, the resultant matrix will be of the size ($n-2$)/2 x ($n-2$)/2 assuming $n$ is even, else if $n$ is odd, then the floor integer value close to $(n-2)/2$.


#### 3. Activation Function

Activation function is very similar to a go or no-go gateway. It is characterized by a threshold value. The output of the neural network goes to the activation function as input, and the activation function after processing them, yields 1 if the output exceeds the threshold, or 0 if found otherwise. Some are capable of mapping the resulting values coninuously in the range [0, 1] or [-1, 1]. Some of the popular activation functions used in neural nets are sigmoid function, softmax, ReLU, and others.

There are two types of activation functions, namely, 
1. Linear activation function, and
2. Non-linear activation function

In the Convolutional Neural Networks (CNNs), Rectified Linear Unit (ReLU) and Softmax activation functions are widely used. The ReLU takes in the output value from neural net output layer and then yields 0 for any resulting negative values leaving out the non-negative results as it is. Thus the output is either 0 or any positive output obtained from the neural net output layer.

The Softmax function is a normalized exponential function. The ouput from the neural network is applied to the exponential and is normalized using the L1 or L2 norm of the exponentials as applied to all the output values.



References:
[1] Wikipedia article on 'Convolution', "[Convolution - Wikipedia](https://en.wikipedia.org/wiki/Convolution)"

[2] Wikipedia article on 'Activation', [Activation function - Wikipedia](https://en.wikipedia.org/wiki/Activation_function)





















