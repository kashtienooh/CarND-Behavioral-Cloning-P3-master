# Behavioral Cloning

### Behavioral Cloning Project

The goals / steps of this project are the following:

*    Use the simulator to collect data of good driving behavior
*    Build, a convolution neural network in Keras that predicts steering angles from images
*    Train and validate the model with a training and validation set
*    Test that the model successfully drives around track one without leaving the road
*    Summarize the results with a written report

### Files Submitted & Code Quality
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

#### My project includes the following files:

* `3_Asigment.ipynb` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model_2_.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results
* `video.mp4` a sample video of the given track with autonomous drive

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

* `python drive.py model_2_.h5`

#### 3. Model parameter tuning

For the parameter tunning the model uses the AdamOptimizer.

#### 4. Appropriate training data

To create the training data I used the training data given by Udacity. I also tried to drive the track one with good behavior one time forward and once backward. The same procedure was done on the track 2.

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My strategy was to use my own data set. At first I collect my data set with the help of the udacity simulator. I ran each track twice in one way and twice in opposite way. The usage of flip technik was not realy helpfull for me.

I use a convolution neural network model similar from the [Nvidia Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I also crop my image to reduce amount of data feeding into RAM.


The final step was to run the simulator to see how well the car was driving around track one. There was one spot where the vehicle fell off the track at the second track I think I should play around with corresction factors. Another way can be so use udacity data set and just add my trimed data set in to it.

At the end of the process, the vehicle is able to drive autonomously around the first track without leaving the road.


#### 2. Nvidia Model Architecture

Here is a visualization of the architecture which based on this paper [Nvidia Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
I crop my images by 50 upper side and 20 from down side of each image. I also use batch normalization between the layers to keep the mean activation near zero. To avoid overfitting I use drop outs flatted fully connected layers. The model used an adam optimizer, so the learning rate was not tuned manually. I used “exponential linear unit” (ELU) which speeds up learning indeep neural networks and leads to higher classification accuracies.


|Layer (type)                 | Output Shape             | Param #  |
|-----------------------------|--------------------------|----------|
|cropping2d_1 (Cropping2D)    |(None, 90, 320, 3)        |0         |        
|lambda_1 (Lambda)            |(None, 90, 320, 3)        |0         |
|batch_normalization_1 (Batch |(None, 90, 320, 3)        |12        |
|conv2d_1 (Conv2D)            |(None, 43, 158, 24)       |1824      |
|conv2d_2 (Conv2D)            |(None, 20, 77, 36)        |21636     |
|conv2d_3 (Conv2D)            |(None, 8, 37, 48)         |43248     |
|batch_normalization_2 (Batch |(None, 8, 37, 48)         |192       |
|conv2d_4 (Conv2D)            |(None, 6, 35, 64)         |27712     |
|conv2d_5 (Conv2D)            |(None, 4, 33, 64)         |36928     |
|dropout_1 (Dropout)          |(None, 4, 33, 64)         |0         |
|flatten_1 (Flatten)          |(None, 8448)              |0         |
|dense_1 (Dense)              |(None, 100)               |844900    |
|batch_normalization_3 (Batch |(None, 100)               |400       |
|dropout_2 (Dropout)          |(None, 100)               |0         |
|dense_2 (Dense)              |(None, 50)                |5050      |
|batch_normalization_4 (Batch |(None, 50)                |200       |
|dropout_3 (Dropout)          |(None, 50)                |0         |
|dense_3 (Dense)              |(None, 10)                |510       |
|batch_normalization_5 (Batch |(None, 10)                |40        |
|dense_4 (Dense)              |(None, 1)                 |11        |

Total params: 982,663
Trainable params: 982,241
Non-trainable params: 422


#### 3. Model parameter tuning

*    The model used an adam optimizer, so the learning rate was not tuned manually.
*    I kept the batch size 32
*    correction factor= 0.2 for adjust left and right camera images
     and ran the model for 12 epochs to make model.py but here just 5 epoch


#### 4. Appropriate training data
I collect my data with Udacity Simulator. I drove each track twice in one way and twice in opposite way. The usage of flip technik was not realy helpfull for me.

#### Summary

To my understanding the way that data set is prepared plays a big role, how it will work on autonomous mode. The amount of data set also play a role. As I had no cuda graphic card and the training is very time costly, I could not play more with all ideas which to make my model work better on the second track. By the way it acts quite good up to last slope! :).