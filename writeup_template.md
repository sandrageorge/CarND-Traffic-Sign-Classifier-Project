#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./writeup_images/histogram.png "Histogram"
[image2]: ./writeup_images/NormalizeImage.png "Normalization"
[image3]: ./writeup_images/0-1.jpg
[image4]: ./writeup_images/11-1.png
[image5]: ./writeup_images/14-1.jpg
[image6]: ./writeup_images/2-1.jpg
[image7]: ./writeup_images/3-1.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sandrageorge/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Using RGB images as input to the network (converting to grayscale will cause loss of important features), RGB images are normalized

RGB image and normalized images

![alt text][image2]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| 1- Convolution 5x5  	| 1x1 stride, valid padding, outputs 28x28x6	|
| 2- RELU				|												|
| 3- Max pooling	    | 2x2 stride,  outputs 14x14x6  				|
| 4- dropout  			| 0.5             								|
| 5- Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16   |
| 6- RELU				|												|
| 7- Max pooling	    | 2x2 stride,  outputs 5x5x6  				    |
| 8- Convolution 5x5	| 1x1 stride, valid padding, outputs 1x1x400    |
| 9- RELU				|												|
| 10- Flatten 7 & 9		| 400x400										|
| 11- Concatenate		| 800   										|
| 12- dropout		    | 0.5             								|
| 13- Fully connected	| 43 classes output								|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 30 epochs, 180 batch size, learing rate 0.001 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 94.4% 
* test set accuracy of 94.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First i tryed to use lenet model but the accuracy was poor.
* What were some problems with the initial architecture?
adapt for RGB images, set the used parameters, adding dropout after convolution.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Adding a dropout after the first convolution layer to decrease the effect of over-fitting.
* Which parameters were tuned? How were they adjusted?
Learning rate of 0.001
Batch size of 180
Epoch count of 30
Keep probability of 0.7

If a well known architecture was chosen: sermanet

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 20 km/h       		| 20 knm/h  									| 
| Right-of way 			| Right-of way									|
| Stop  				| Stop											|
| 50 km/h	      		| 30 km/h					 				    |
| 60 km/h	      		| 50 km/h					 				    |


The model was not able to detect 50 and 60 km/h sign.