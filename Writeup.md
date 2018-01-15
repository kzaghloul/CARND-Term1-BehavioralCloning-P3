
# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: Network_Visualization.png "Model Visualization"
[image2]: center_driving.jpg "Center Driving"
[image3]: Sharp1.jpg "Sharp1 Image"
[image4]: Sharp2.jpg "Sharp2 Image"
[image6]: plot.png "Loss plot Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model follows the NVIDIA self driving car architecture and consists of a convolution neural network with 5x5 and 3x3 filter sizes (model.py lines 50-67) 

The model includes RELU layers to introduce nonlinearity (code lines 58, 60-63), and the data is normalized in the model using a Keras lambda layer (code line 56). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 59). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 4-11). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and extra sharp turning.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the car to self drive around track one only.

My first step was to use a convolution neural network model similar to the NVIDIA Self Driving Car architecture. I thought this model might be appropriate because it yielded the most accurate and relatively fast results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there is a dropout of half the data after the first convolution layer.

Then I applied several 2d convultion layers with a RELU activation, followed by a linear flattening and reduction in size.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, the sharp turn following the exit of the bridge. To improve the driving behavior in these cases, I recorded extra turning data for it.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 55-72) consisted of a convolution neural network.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded extra turning data of the vehicle taking the sharper turns:

![alt text][image3]
![alt text][image4]


To augment the data sat, I also flipped images and angles thinking that this would help generalize the model better and teach it to turn to the right better.

After the collection process, I had almost 14000 data points. I then preprocessed this data by splitting it into trainning and validation sets and normalized using a Keras Lambda layer.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the plot below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image6]

## Future Improvements:
1.Recovery driving data

2.counter clock wise lap

3.track 2 data

### References:
1.Udacity lessons and code
2.NVIDIA Deep Learning Self Driving Car code architecture
