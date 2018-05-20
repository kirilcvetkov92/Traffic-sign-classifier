## Project: Build a Traffic Sign Recognition Program
---


The Project
---
The goals/steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:
  * tensorflow-gpu==1.7.0
  * scipy==1.0.0
  * matplotlib==2.0.0
  * numpy==1.14.2
  * opencv-contrib-python==3.4.0.12
  * sklearn==0.18.2


### Dataset
[Download the data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip).

### Load the data set and basic summary
After loading the dataset, I got the following summary information:
* Number of training examples is: 34799
* Number of testing examples is: 12630
* Number of validation examples is: 4410
* Image shape is: (32 32, 3)
* Number of classes labels is: 43

#### Exploratory visualization
This grid of images is representing one chosen random image from of each class from the training set
![Exploratory visualization](/documentation/data.png)

**Distribution**

Now we are going to explore the distribution and take look at the distribution of classes in the training, validation and test set.

From the histograms below, we can clearly see that the distribution of train, validation, and test set is nearly the same, 
but the problem is that there is a huge variability of the distribution between class instances within the dataset, 
and we can further investigate whether it can cause some problems during our training, 
and maybe we can develop augmentation techniques to equalize them.

![Distribution](/documentation/graphs.png)


Maximum class labels instances in train data 2010.0
Minimum class labels instances in test data 180.0 

Maximum class labels instances in validation data 240.0
Mininum class labels instances in validation data 30.0 

Maximum class labels instances in test data 750.0
Minimum class labels instances in test data 60.0

#### Data preprocess

I performed only small preprocessing technique which was quick win form me and enough good for my model. 
For image data: pixel = pixel/ 255 which was a quick way that normalizes the points between 0 and 1 which fit well with the activation function expected input.

***Data Augmentation***

The first thing I tried is to augment the data replicating the class labels which are rare in the dataset, so it can reduce the high variance of our model (Overfitting)

The augmentation techniques I tried were combinations of : 

* center zoom
* sharpening 
* Contrast
* translation
* rotation
* salt and pepper noise
    
***Conclusion : I realized that data augmentation cannot make drastic improvements to the performance of my model, and the augmentation step was omitted due to slowing down the entire training procedure***

### Model Architecture
Below are the details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are also included 

***Inception modules***

Inspired by GoogleNet model (which was made by the foundation of DeepDream) I used 2 ***Inception modules*** in my neural network.
It's basically stacking multiple pooling and convolution blocks, see on the picture below : 
![Distribution](/documentation/inception.png)

To learn more about inception modules click [here](https://www.youtube.com/watch?v=VxhSouuSZDY).

***Regularization***

Success: 

* ***Dropout***: I used dropout only for the fully connected layers, which took me to the best performance for my training steps

Failure :
* ***L2 regularization***: Tried but didn't improve anything ***Omitted in the final model***
* ***Batch normalization***: Tried but didn't improve anything ***Omitted in the final model***

***Initializer***
Xavier weight initializer was used for this model

***Activation functions***
*  Relu

***Optimizer***
* Adam optimizer

***Model sizes***

| Layer         		|     Description	        					| Input     | Output      |
|:---------------------:|:---------------------------------------------:|:---------:|:-----------:| 
| Convolution       	| scope:conv0; kernel: 7x7; stride:2x2; padding: Same; output_size=64  	    | (?,32,32,3)   | (?,16,16,64)     |
| Inception  	      	| scope:Inception3a            	    | (?,16,16,64)    | (?,16,16,256)      |
| Inception       	|  scope:Inception3b	    |  (?,16,16,256)    | (?,16,16,480)     |
| Max pooling	      	|scope:pool2; kernel: 3x3; stride:2x2; padding: Same; output_size=64 		| (?,16,16,480) | (?,8,8,480)      |
| Flatten				| Squeeze the cube into one dimension			| (?,8,8,480)      (?,30720)          |
| Fully connected		| scope:fully_2; pairwise connections between all nodes	    | (?,30720)       | (?,200)         |
| Fully connected		|  scope:fully_3; pairwise connections between all nodes		        | (?,400)       | (?,400)         |
| Fully connected		|  scope:fully_4; pairwise connections between all nodes	  		| (?,400)           | (?,300)             |
| Fully connected		|  scope=logits; pairwise connections between all nodes	  		| (?,300)          | (?,42)            |


***Visualization of the model ***
![Model architecture](/documentation/model.png)
