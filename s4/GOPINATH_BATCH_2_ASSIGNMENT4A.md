# Assignment 4A

#### *Batch 2 -- Gopinath Venkatesan *

#### Objective: 
With the given MNIST classification CNN code, achieve a validation accuracy more than what is achieved with the current keras sequential model but using lesser model parameters.

##### Attempt 1:
Without using 5 x 5 filter and MaxPooling (to reduce the output size and hence the model parameters)

Here I used 20 layers of filter 3 x 3 as a 2nd layer pushing the previous 10 layers of 1 x 1 to the third layer.

The code for the same is given below:

```
from keras.layers import Activation
model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
# This gives an output of 26 x 26 x 32, which feeds into the next layer as input
model.add(Convolution2D(20, 3, 3, activation='relu'))
# This yields 24 x 24 x 20
model.add(Convolution2D(10, 1, activation='relu'))
# This gives an output of 24 x 24 x 10
model.add(Convolution2D(10, 24))
# The final output from the above convolution is 1 x 1 x 10
model.add(Flatten())
model.add(Activation('softmax'))
```

The model summary for the above model is given below:

1st Layer - Conv2D using 32 3 x 3 filters - Outputs 26 x 26 x 32 with 320 parameters.

2nd Layer - Conv2D using 20 3 x 3 filters - outputs 24 x 24 x 20 with 5780 parameters.

3rd Layer - Conv2D using 10 1 x 1 filters outputs 24 x 24 x 10 with 210 parameters.

4th layer - Fully connected Conv2D layer outputs 1 x 1 x 10 with 57610 parameters.

Total params: 63920
Trainable params: 63920
Non-trainable params: 0

**Validation Accuracy: 98.6%**

Original validation accuracy of the model before the introduction of 20 3 x 3 filters is 98.29%.

##### Attempt 2:

In the second attempt, I used 5 x 5 filter instead of the earlier 3 x 3 filters but kept the number of filters as the same i.e., 32 filters of size 5 x 5.

Another major difference is that I used a MaxPooling layer to reduce the image size in the 3rd layer from 22 x 22 x 20(channels) to a size of 11 x 11 x 20(channels) using the 2 x 2 pooling matrix with a stride of 2. This considerably reduced the number of model parameters.

```
from keras.layers import Activation
model = Sequential()

model.add(Convolution2D(32, 5, 5, activation='relu', input_shape=(28, 28, 1)))
# This gives an output of 24 x 24 x 32, which feeds into the next layer as input
model.add(Convolution2D(20, 3, 3, activation='relu'))
# This yields 22 x 22 x 20
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None))
# This adds a MaxPool layer so as to reduce the size from 22 x 22 x 20 to 11 x 11 x 20
model.add(Convolution2D(10, 3, 3, activation='relu'))
# This yields 9 x 9 x 10
model.add(Convolution2D(10, 1, activation='relu'))
# This gives an output of 9 x 9 x 10
model.add(Convolution2D(10, 9))
# The final output from the above convolution is 1 x 1 x 10
model.add(Flatten())
model.add(Activation('softmax'))
```

These model summary details obtained for this model is given below:

1st Layer - Conv2D using 32 5 x 5 filters - Outputs 24 x 24 x 32 with 832 model parameters.

2nd Layer - Conv2D using 20 3 x 3 filters - Outputs 22 x 22 x 20 with 5780 parameters.

3rd Layer - MaxPooling2D using 2 x 2 filter with a stride of 2. Since this is a pooling operation, no parameters were used.

4th Layer - Conv2D using 10 3 x 3 filters - Outputs 9 x 9 x 10 with 1810 parameters.

5th Layer - Conv2D using 10 1 x 1 filters - Outputs 9 x 9 x 10 with 110 parameters.

6th Layer - Conv2D using 10 9 x 9 filters - Outputs 1 x 1 x 10 with 8110 parameters.

Total params: 16642
Trainable params: 16642
Non-trainable params: 0

**Validation Accuracy: 98.89%**

This is better than the previous attempt (98.6%) and the original validation accuracy of 98.29%

I have not tried adjusting the batch-size, number of epochs.


