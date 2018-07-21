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
[image3]: ./examples/distribution_new_testset.png "New Testset Visualization"
[image4]: ./test/2.jpg "Speed limit 50 km/h"
[image5]: ./test/12.jpg "Priority road"
[image6]: ./test/13.jpg "Yield"
[image7]: ./test/14.jpg "Stop"
[image8]: ./test/33.jpg "Right turn only"
[image9]: ./examples/accuracy.png "Validation Accuracy"
[image10]: ./examples/error.png "Validation Error""
[image11]: ./examples/training_accuracy.png "Training Accuracy"
[image12]: ./examples/training_error.png "Training Error"

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

Because of the unequally distribution of training data, I used some easy methods to generate new images. I applied gamma correction from -5 and +5, rotation of 5 to 355 degrees and shifting the image to all directions. I repeated this process until each traffic sign in the training set reached at least 1000 images:

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.
I played around with image grayscaling like we did in the lane-finding-project, but it wasn't helpful to achieve a better accuracy. So I decided just to use the normalization technique, proposed by you, to work with images of the same size and same colour regions.

![alt text][image2]

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
First, I decided for 40 epochs, as this was after several runs, the number which brought a good result. With more epochs, the accuracy didn't raise anymore. But to show the learn-rate, I run it with 100 epochs in the end.
The batch_size of 64 was also a good value, to see progress on the AWS after a few seconds, while using the power of 2.
The learn_rate was chosen, after I tried 0.01 and 0.0001, where both brought a bad result.

To prevent from overfitting, I added the L2 regularization, but it didn't boost up the validation accuracy pretty much. afterwards, I figured out, that a little higher SIGMA=0.15 for the LeNet, brought a more stable result by having less errors.

Compared to the Adam Optimizer, the Stochastic Gradient Descend had a bad performance. The increase of the accuracy was very slow, especially with a learn-rate of 0.001. With a learn rate of 0.01, the accuracy reached only 93,1% after 100 epochs. I didn't wait for the result of a learn-rate of 0.001, as the accuracy was still at 85% after 89 Epochs.
The Adam Optimizer had a fast increase and a good result with a learn-rate of 0.01 and 0.001. 

I also visualized the learning rate:
![alt text][image9]
![alt text][image10]

#### 4. The approach taken for finding a solution 

I chose the lenet architecture, because it was a solid starting point for this limited image recognition. The easiest way with a very good result was to tune the activation functions by adding a dropout function, to prevent from overfitting. First, I tried to just add one dropout to the first activation function, but I figured out, that it wasn't bad, to put it on multiple RELUs.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.944
* test set accuracy of 0.934

### Test a Model on New Images

#### 1. Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

After about 50 rounds of training with different parameters, I can say, that the most mistakes occur with red sign (stop and speed limit).

Additionally, in the last round I added more pictures (18) from Google, to see, if my tuned parameters didn't bleed into the network. 

#### 2. The model's predictions on these new traffic signs 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 20 km/h	      		| 20 km/h						 				|
| No passing 3.5 tons	| No passing 3.5 tons			 				|
| 120 km/h	      		| WRONG: 100 km/h				 				|
| Turn right ahead		| Turn right ahead								|
| 60 km/h	      		| WRONG: 20 km/h				 				|
| Yield					| Yield											|
| Stop Sign      		| Stop		  									| 
| 30 km/h	      		| 30 km/h						 				|
| 50 km/h	      		| 50 km/h						 				|
| No entry	      		| No entry						 				|
| 70 km/h	      		| 70 km/h						 				|
| 100 km/h	      		| 100 km/h						 				|
| No passing      		| WRONG: 70 km/h				 				|
| No vehicles      		| No vehicles					 				|
| Priority road     	| Priority Road 								|
| 80 km/h	      		| WRONG: 20 km/h				 				|
| End of (80km/h)	  	| End of speed limit (80km/h)	 				|
| Right-of-way next int.| Right-of-way at the next intersection			|


The model was able to correctly guess 14 of the 18 traffic signs, which gives an accuracy of 77%. During the development, I achieved everything between 0-100%. 

After adding the L2 regularization, the accuracy of my new images went down from 80% to 60%, although the test and validation accuracy of the initial dataset grew a little bit. 
On the one hand, the network is not quite perfect, but on the other hand, it is very interesting. Although I used exactly the same training and validation set, the prediction some how differs at different points.

#### 3. Top 5 Softmax probabilities

In order to understand the prediction of the new test images, we can see the top 5 softmax predictions for each image here. As you see in the following table, the predictions are pretty sure except of one.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999		      		| 20 km/h						 				|
| 1.0					| No passing 3.5 tons			 				|
| 1.0		      		| WRONG: 100 km/h				 				|
| 0.999					| Turn right ahead								|
| 0.999		      		| WRONG: 20 km/h				 				|
| 1.0					| Yield											|
| 0.985		      		| Stop		  									| 
| 0.945	      			| 30 km/h						 				|
| 0.999	      			| 50 km/h						 				|
| 1.0		      		| No entry						 				|
| 0.998	      			| 70 km/h						 				|
| 1.0		      		| 100 km/h						 				|
| 0.998      			| WRONG: 70 km/h				 				|
| 0.836		      		| No vehicles					 				|
| 1.0			     	| Priority Road 								|
| 0.999	      			| WRONG: 20 km/h				 				|
| 0.994				  	| End of speed limit (80km/h)	 				|
| 1.0					| Right-of-way at the next intersection			|

Looking at the predicition and indices arrays of the top 5 values, we can see the following:
[[  9.99968052e-01   3.10057585e-05   9.23682023e-07   1.13195417e-08
    2.25189289e-09]
 [  1.00000000e+00   7.68846347e-18   6.53288966e-24   2.34391036e-25
    1.16673341e-25]
 [  1.00000000e+00   3.26684652e-10   2.68439124e-14   4.87774331e-22
    6.10197242e-24]
 [  9.99995708e-01   4.34074718e-06   3.79451226e-09   7.40946055e-11
    5.97802016e-11]
 [  9.99986649e-01   1.33117201e-05   2.20733920e-08   1.56071697e-10
    2.87977519e-12]
 [  1.00000000e+00   1.66389351e-20   2.69127170e-21   2.44133265e-22
    6.64646173e-23]
 [  9.84829783e-01   1.11954743e-02   3.01783951e-03   9.40331724e-04
    1.16979472e-05]
 [  9.44961667e-01   4.15626131e-02   1.34559460e-02   9.94019774e-06
    9.67882352e-06]
 [  9.99596894e-01   3.45071283e-04   5.19990426e-05   6.08344180e-06
    1.75850428e-08]
 [  1.00000000e+00   1.43158241e-14   2.18692705e-15   2.17163895e-15
    1.32042558e-15]
 [  9.98651206e-01   1.34881947e-03   1.10157528e-08   2.59621280e-09
    8.01834554e-12]
 [  1.00000000e+00   9.69772654e-17   7.38448907e-23   6.47705093e-26
    5.96129051e-28]
 [  9.98916388e-01   5.41046727e-04   5.06571669e-04   2.29831167e-05
    1.23580530e-05]
 [  8.36316884e-01   1.60951361e-01   1.58869568e-03   1.13729585e-03
    2.37096879e-06]
 [  1.00000000e+00   3.00644043e-09   1.49217491e-14   6.43345198e-15
    8.02986334e-16]
 [  9.99809921e-01   1.58701892e-04   3.12073389e-05   2.70752395e-07
    1.09160458e-08]
 [  9.94092643e-01   5.89046068e-03   1.58133244e-05   4.33851767e-07
    4.30774463e-07]
 [  1.00000000e+00   1.25704475e-15   6.06870890e-16   4.46099695e-16
    1.34858061e-16]]

[[ 0  1 27 16 39]
 [10  9 17 16 35]
 [ 7  8  5 16  4]
 [33 35 39 26 18]
 [ 0 27  3  2  4]
 [13 29 26 15 28]
 [14  0 24  4 26]
 [ 1 27  0  6 16]
 [ 2  1  5  4  0]
 [17 10 12 14  0]
 [ 4  0 27 18 14]
 [ 7  5 16  8  4]
 [ 4  7  9 16 14]
 [15 41 38  4  2]
 [12 17 36 42  5]
 [ 0  1 27 40  6]
 [ 6 32 42 39 41]
 [11 28 20 30 16]]

I also added a visualization of the predicted signs, showing the accuracy of the prediction. This is good to understand, what the net predicted.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### I was not able to finish that exercise, due to less time.


