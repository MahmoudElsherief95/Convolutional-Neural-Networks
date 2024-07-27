# Convolutional-Neural-Networks

In this assignment, the goal is to get familiar with **Convolutional Neural Networks**. Essentially, a CNN is a multi-layer perceptron that uses convolutional instead of fully connected layers. Since convolutions are known to be useful for image processing, CNNs have become a powerful tool for learning features from images. However, they have proven to beat alternative architectures in a variety of other domains.

<img width="1593" alt="image" src="https://github.com/user-attachments/assets/71d44c59-2aea-4608-bc61-b9c8129a9b47">

### Exercise 1: Cross-correlation vs Convolution

Implementation-wise, there is little difference between cross-correlation and convolution. It is even quite straightforward to implement one, given an implementation of the other. To keep things simple, this exercise is limited to the one-dimensional variants of these operations (for now). How hard would it be to make your implementation of the convolution function commutative?

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 2: Multi-channel Convolutions 

Time to implement an actually practical convolution function that can handle multiple channels. Let us make it a 2D convolution at once.

 > Implement the `multi_channel_convolution2d` function below. You can use the `sig2col` function to implement the convolution by means of a dot product.
 
**Hint:** When using the `sig2col` function, you might need to fiddle with the order of dimensions of your numpy arrays to align everything properly.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
<img width="1619" alt="image" src="https://github.com/user-attachments/assets/db364fd8-6fdd-41fc-83e5-607208241dc2">

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 3: Convolutional Layer 

Now, you should be able to implement both forward and backward pass in a module. Have you already thought about the shape of the bias parameter?

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 4: Some Linear Units (3 Points)

A deep learning framework would not be complete without the ReLU and ELU activation functions. Time to add them!

 > Implement the `ReLU` and `ELU` activation function modules.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Spatial Reduction

The *weight sharing* in convolutional neural networks can drastically reduce the memory requirements for the weights. This effectively allows the input data to become larger, but since we need to store parts of the forward pass for back-propagation, the gains are rather limited. Of course, standard convolutions reduce the spatial dimensions, but this linear reduction is often too slow to counter the increased memory requirements due to network depth.

###### Pooling

In order to make working with big images feasible, we need techniques to reduce the spatial dimensions more strongly. This is where *pooling* layers prove very useful. A pooling layer reduces the spatial dimensions by combining a window of pixels to a single pixel. By sticking a pooling layer after every convolutional layer, the spatial dimensions are reduced exponentially, rather than linearly. This allows convolutional neural networks to process big chunks of data.

There are different ways to summarise multiple pixels into a single pixel. Two very common pooling techniques are

 1. **Average pooling** replaces the pixels by the mean intensity value in the window. 
 2. **Max pooling** replaces the pixels by the maximum intensity in the window.

###### Strides

In modern convolutional neural networks, *strided* or *dilated* convolutions (see visualisations below) are often preferred over pooling. With strided convolutions, the windows are shifted The main advantage of strided or dilated convolutions over pooling is that they can be learnt. This means that instead of relying on a fixed pooling technique, it is possible to effectively learn how the pixels in the window are to be summarised. Also note that average pooling can indeed be represented as a strided convolution with weights $\frac{1}{\text{window size}}$.

<div style="text-align: center">
  <figure style="display: inline-block; width: 49%;">
    <img style="padding: 46px 50px" src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_strides.gif" />
    <figcaption style="width: 100%;"> Strided convolution </figcaption>
  </figure>
  <figure style="display: inline-block; width: 49%;">
    <img src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/dilation.gif" />
    <figcaption style="width: 100%; text-align: center;"> Dilated convolution </figcaption>
  </figure>
</div>

*visualisations taken from the [github repo](https://github.com/vdumoulin/conv_arithmetic) that comes with [this guide](https://arxiv.org/abs/1603.07285)*

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 5: Pooling (5 Points)

Since the `sig2col` function provides an array with the window elements in each column, it can also be used to implement pooling layers, when used correctly.



