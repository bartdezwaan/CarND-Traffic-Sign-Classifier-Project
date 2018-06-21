# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./output_images/bar_chart_visualization.png "Visualization"
[image2]: ./output_images/to_gray_scale.png "Grayscaling"
[image3]: ./output_images/histogram_equalization.png "Histogram Equalization"
[image4]: ./output_images/augmentation.png "Image Augmentation"
[image5]: ./output_images/traffic_signs.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Summary of the data set.

The code for this part can be found in the 2th till 4th cell of the notebook (Traffic_Sign_Classifier.ipynb)

Numpy is used to calculate summary statistics of the traffic signs data set and matplotlib to visualize it with a bar chart:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The dataset is not evenly distibuted.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing that the data is not evenly distributed

![visual][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The code for this part can be found in the 5th till 13th cell of the notebook (Traffic_Sign_Classifier.ipynb)

I converted the images to grayscale. Eyeballing it, it looked like edges and figures where better detectable. Testing it when training the network confirmed that converting to grayscale worked better.

Here is an example of a traffic sign image before and after grayscaling.

![grayscaling][image2]

After grayscaling I improved contrast through histogram equalization. This gave me the biggest improvement in accuracy (around 1%)

Here is an example of a traffic sign image before and after histogram equalization

![histogram equalization][image3]

I also normalized the data because machine learning algorithms tend to find it easier when numbers fall within a small range.

I decided to not generate additional data. I experimented with data augmentation, because the dataset is inbalanced. I generated extra images for classes that had fewer then 2000 images. I ended up with all classes having roughly 2000 images. This however decreased performance a lot. Probably augmentating under presented classed to much does not help.

For completion, here is an example of augmented images:

![alt text][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128 				|
| Flatten	        	| outputs 3200 				|
| Fully connected		| output 120        									|
| RELU					|											|
| Dropout       		| dropout 80% Keep        									|
| Fully connected		| output 84        									|
| RELU					|												|
| Fully connected		| output 43        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for this part can be found in the 14th and 15th cell of the notebook (Traffic_Sign_Classifier.ipynb)

I tested with different parameters when testing the model.
The parameters that gave me the best results are:

* Epochs: 25
* Batch size: 128
* Rate: 0.0008

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.999%
* validation set accuracy of 0.979%
* test set accuracy of 0.963%

I started out with the basic LeNet architecture and started to improve from there:
The following steps where taken:

* I experimented with the output depths of the convolutional layers. Increasing the depth gave me better accuracy
* I also tried adding more convolutional layers and/or fully connected layers. This decreased my accuracy
* Dropout was added to prevent overfitting. Strangly this did not really improve performance. If I added more then one dropout layer, it decreased accuracy.
* I tried quit some Epoch, batch size and rate settings, changing them one at a time

The final network seems to perform quit well, since the train, validation and test accuracy are not to far appart from each other.

The LeNet architecure seems like a good starting point since it performence well on the MNIST dataset, which also has small images.
Since the task we are trying to accomplish is not to complex, a deeper neural network does make things harder. It will take longer to train and preventing overfitting is a bigger challange.

I did try the VGG16 Network and use transfer learning to improve accuracy. I was not able to improve on my accuracy from the LeNet based architecture.

### Test a Model on New Images

The code for this part can be found in the 16th and 17th cell of the notebook (Traffic_Sign_Classifier.ipynb)

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![traffic signs][image5]

The 2th and 5th images are not completly centered. This might make it more difficult to classify them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for this part can be found in the 19th and 20th cell of the notebook (Traffic_Sign_Classifier.ipynb)

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery Road			| Slippery Road      							|
| 30 km/h	      		| Yield					 				|
| U-turn     			| U-turn 										|
| Road work					| Road Work											|
| Stop Sign      		| Stop sign   									| 


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is a bit dissapointing.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image the probabilities where:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.			| Slippery Road      							|
| 0.	      		| Speed limit (20km/h)					 				|
| 0.     			| Speed limit (30km/h) 										|
| 0.				| Speed limit (50km/h)											|
| 0.      		| Speed limit (60km/h)   									| 

For the second image the probabilities where:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.			| Yield      							|
| 0.	      		| Speed limit (20km/h)					 				|
| 0.     			| Speed limit (30km/h) 										|
| 0.				| Speed limit (50km/h)											|
| 0.      		| Speed limit (60km/h)   									| 

For the third image the probabilities where:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.			| Road work      							|
| 0.	      		| Speed limit (20km/h)					 				|
| 0.     			| Speed limit (30km/h) 										|
| 0.				| Speed limit (50km/h)											|
| 0.      		| Speed limit (60km/h)   									| 

For the fourth image the probabilities where:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.			| Stop      							|
| 0.	      		| Speed limit (20km/h)					 				|
| 0.     			| Speed limit (30km/h) 										|
| 0.				| Speed limit (50km/h)											|
| 0.      		| Speed limit (60km/h)   									| 

For the second fifth the probabilities where:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.			| Dangerous curve to the right      							|
| 0.	      		| Speed limit (20km/h)					 				|
| 0.     			| Speed limit (30km/h) 										|
| 0.				| Speed limit (50km/h)											|
| 0.      		| Speed limit (60km/h)   									| 

