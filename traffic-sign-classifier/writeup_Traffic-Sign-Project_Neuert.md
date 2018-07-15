# **Traffic Sign Recognition** 

## Writeup

### Marcus Neuert, July 11 2018

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distribution_testset.png "Visualization"
[image2]: ./examples/normalization.png "Normalization"
[image4]: ./test/2.jpg "Speed limit 50 km/h"
[image5]: ./test/12.jpg "Priority road"
[image6]: ./test/13.jpg "Yield"
[image7]: ./test/14.jpg "Stop"
[image8]: ./test/33.jpg "Right turn only"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. My project files on a git repository.

You're reading it! and here is a link to my [project code](https://github.com/MaggiN1337/carnd-term1/blob/master/traffic-sign-classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 in RGB colors
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the test data is distributed over the different traffic signs. The distribution of the validation and test look almost identically, because the number of images is nearly equalent to the images of the test set. That's a good precondition for the training.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.
I played around with image grayscaling like we did in the lane-finding-project, but it wasn't helpful to achieve a better accuracy. So I decided just to use the normalization technique, proposed by you, to work with images of the same size and same colour regions.

![alt text][image2]

I'd love to generate new test data by blacking out 40% of the upper or lower image or just overlight some parts, in order to simulate shadow or sun reflections. But due to time problems, I skipped this step.

#### 2. Final model architecture 

I decided to use the structure of LeNet with only a few modifications. Mainly I added Dropouts to all Activation, as it optimzed the accuracy significantly. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Dropout & RELU		| The results were 2% better with using dropout	|
| Max pooling	      	| 2x2 stride, same padding, outputs 14x14x6 	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Dropout & RELU		| 												|
| Dropout & Max pooling	| 2x2 stride, same padding, outputs 5x5x16		|
| Flatten				| outputs 5x5x16 = 400        					|
| Fully connected		| outputs 120        							|
| Dropout & RELU		| 												|
| Fully connected		| outputs 84        							|
| Dropout & RELU		| 												|
| Fully connected		| outputs 43 (number of classes)        		|
|						|												|
 


#### 3. Training model

I moved the parameters to the top of the code, in order to configure everything at the beginning.
To train the model, I used a lean_rate of 0.001, a batch size of 128 and 100 epochs.
First, I decided for 40 epochs, as this was after several runs, the number which brought a good result. With more epochs, the accuracy didn't raise anymore. But to show the learn-rate, I run it with 100 epochs in the end.
The batch_size of 64 was also a good value, to see progress on the AWS after a few seconds, while using the power of 2.
The learn_rate was chosen, after I tried 0.01 and 0.0001, where both brought a bad result.

To prevent from overfitting, I added the L2 regularization, but it didn't boost up the validation accuracy pretty much.

Compared to the Adam Optimizer, the Stochastic Gradient Descend had a bad performance. The increase of the accuracy was very slow, especially with a learn-rate of 0.001. With a learn rate of 0.01, the accuracy reached only 93,1% after 100 epochs. I didn't wait for the result of a learn-rate of 0.001, as the accuracy was still at 85% after 89 Epochs.
The Adam Optimizer had a fast increase and a good result with a learn-rate of 0.01 and 0.001. 

#### 4. The approach taken for finding a solution 

I chose the lenet architecture, because it was a solid starting point for this limited image recognition. The easiest way with a very good result was to tune the activation functions by adding a dropout function, to prevent from overfitting. First, I tried to just add one dropout to the first activation function, but I figured out, that it wasn't bad, to put it on multiple RELUs.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.957
* test set accuracy of 0.947

I think, to get a higher accuracy of the validation set, it is not really necessary to tune the neural network. The better way would be, to add more training and validation data in the way I mentioned above.

### Test a Model on New Images

#### 1. Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The stop sign image might be difficult to classify because it looks a bit dirty in the red area and it has some small shadow on the top right corner.

#### 2. The model's predictions on these new traffic signs 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right turn only		| Right turn only								|
| Yield					| Yield											|
| Stop Sign      		| 30 km/h	  									| 
| 50 km/h	      		| 30 km/h						 				|
| Priority road     	| Priority Road 								|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. During the development, I achieved 100% once. In the next run, the neural networks predicted the 50 km/h as a 30 km/h sign. 
After adding the L2 regularization, the accuracy of my new images went down from 80% to 60%, although the test and validation accuracy of the initial dataset grew a little bit. 
On the one hand, the network is not quite perfect, but on the other hand, it is very interesting. Although I used exactly the same training and validation set, the prediction some how differs at different points.

#### 3. Top 5 Softmax probabilities

In order to understand the prediction of the new test images, we can see the top 5 softmax predictions for each image here. As you see in the following table, the predictions are pretty sure except of one.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99     				| Right turn only 								|
| 1.0					| Yield											|
| 0.87         			| WRONG: 30 km/h instead of Stop sign  			| 
| 0.99	      			| WRONG: 30 km/h instead of 50km/h				|
| 1.0				    | Priority Road      							|

Looking at the predicition and indices arrays of the top 5 values, we can see the following:
 [  9.99988e-01   7.12141e-06   2.72286e-06   1.27321e-06   7.10994e-07]
 [33 40 11 18 39]
 
 [  1.00000e+00   2.97787e-16   6.79328e-17   9.42291e-18   2.85270e-18]
 [13 14  1 15 17]
 
 [  8.70176e-01   9.06256e-02   2.25370e-02   1.43961e-02   8.17040e-04]
 [ 1 14  4 13 38]
 
 [  9.99998e-01   2.16475e-06   1.26999e-09   7.27014e-11   2.88481e-11]
 [ 1  2  4 21  0]
 
 [  1.00000e+00   4.96219e-09   3.45306e-09   1.93385e-09   3.79191e-10]
 [12 32 13 14 38]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### I was not able to finish that exercise, due to less time.


