# ERA S6 Assignment - Part 2 (MNIST Dataset)

## Target
 - 99.4% validation accuracy
 - Less than 20k Parameters
 - Less than 20 Epochs

## We can use below concept to achieve the target
How many layers,\
MaxPooling,\
1x1 Convolutions,\
3x3 Convolutions,\
Receptive Field,\
SoftMax,\
Learning Rate,\
Kernels and how do we decide the number of kernels?\
Batch Normalization,\
Image Normalization,\
Position of MaxPooling,\
Concept of Transition Layers,\
Position of Transition Layer,\
DropOut\
When do we introduce DropOut, or when do we know we have some overfitting\
The distance of MaxPooling from Prediction,\
The distance of Batch Normalization from Prediction,\
When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)\
How do we know our network is not going well, comparatively, very early\
Batch Size, and effects of batch size\

## Model Summary

``` ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 128, 28, 28]           1,280
       BatchNorm2d-2          [-1, 128, 28, 28]             256
         Dropout2d-3          [-1, 128, 28, 28]               0
            Conv2d-4            [-1, 8, 30, 30]           1,032
            Conv2d-5           [-1, 16, 30, 30]           1,168
       BatchNorm2d-6           [-1, 16, 30, 30]              32
         Dropout2d-7           [-1, 16, 30, 30]               0
         MaxPool2d-8           [-1, 16, 15, 15]               0
            Conv2d-9           [-1, 16, 15, 15]           2,320
      BatchNorm2d-10           [-1, 16, 15, 15]              32
        Dropout2d-11           [-1, 16, 15, 15]               0
           Conv2d-12           [-1, 32, 15, 15]           4,640
      BatchNorm2d-13           [-1, 32, 15, 15]              64
        Dropout2d-14           [-1, 32, 15, 15]               0
        MaxPool2d-15             [-1, 32, 7, 7]               0
           Conv2d-16             [-1, 16, 9, 9]             528
           Conv2d-17             [-1, 16, 9, 9]           2,320
      BatchNorm2d-18             [-1, 16, 9, 9]              32
        Dropout2d-19             [-1, 16, 9, 9]               0
           Conv2d-20             [-1, 32, 9, 9]           4,640
      BatchNorm2d-21             [-1, 32, 9, 9]              64
        Dropout2d-22             [-1, 32, 9, 9]               0
           Conv2d-23           [-1, 10, 11, 11]             330
        AvgPool2d-24             [-1, 10, 1, 1]               0
================================================================
Total params: 18,738
Trainable params: 18,738
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 3.08
Params size (MB): 0.07
Estimated Total Size (MB): 3.15
----------------------------------------------------------------
```

## Test set: Average loss: 0.0190, Accuracy: 9939/10000 (99.4%) 
