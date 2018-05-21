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
This project requires:
  * tensorflow-gpu==1.7.0
  * scipy==1.0.0
  * matplotlib==2.0.0
  * numpy==1.14.2
  * opencv-contrib-python==3.4.0.12
  * sklearn==0.18.2

## Dataset Exploration

### Dataset Summary

[Download the data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip).

#### Load the data set and basic summary
After loading the dataset, I got the following summary information:
* Number of training examples is: 34799
* Number of testing examples is: 12630
* Number of validation examples is: 4410
* Image shape is: (32 32, 3)
* Number of classes labels is: 43

### Exploratory visualization
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

## Design and Test a Model Architecture

### Preprocessing

I performed only small preprocessing technique which was quick win form me and enough good for my model. 
For image data: pixel = pixel/ 255 which was a quick way that normalizes the points between 0 and 1 which fit well with the activation function expected input.

#### Data Augmentation

The first thing I tried is to augment the data replicating the class labels which are rare in the dataset, so it can reduce the high variance of our model (Overfitting)

The augmentation techniques I tried were combinations of : 

* center zoom
* sharpening 
* Contrast
* translation
* rotation
* salt and pepper noise
    
***Conclusion: I realized that data augmentation cannot make drastic improvements to the performance of my model, and the augmentation step was omitted due to slowing down the entire training procedure***

### Model Architecture
Below are the details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are also included 

#### Inception modules

Inspired by GoogleNet model (which was made by the foundation of DeepDream) I used 2 ***Inception modules*** in my neural network.
It's basically stacking multiple pooling and convolution blocks, see on the picture below : 
![Distribution](/documentation/inception.png)

To learn more about inception modules click [here](https://www.youtube.com/watch?v=VxhSouuSZDY).

#### Regularization

Success: 

* ***Dropout***: I used dropout only for the fully connected layers, which took me to the best performance for my training steps

Failure :
* ***L2 regularization***: Tried but didn't improve anything ***Omitted in the final model***
* ***Batch normalization***: Tried but didn't improve anything ***Omitted in the final model***

***Initializer***
* Xavier weight initializer was used for this model

***Activation functions***
*  Relu

***Optimizer***
* Adam optimizer

#### Model sizes

| Layer         		|     Description	        					| Input     | Output      |
|:---------------------:|:---------------------------------------------:|:---------:|:-----------:| 
| Convolution       	| scope:conv0; kernel: 7x7; stride:2x2; padding: Same; output_size=64  	    | (?,32,32,3)   | (?,16,16,64)     |
| Inception  	      	| scope:Inception3a            	    | (?,16,16,64)    | (?,16,16,256)      |
| Inception       	|  scope:Inception3b	    |  (?,16,16,256)    | (?,16,16,480)     |
| Max pooling	      	|scope:pool2; kernel: 3x3; stride:2x2; padding: Same; output_size=64 		| (?,16,16,480) | (?,8,8,480)      |
| Flatten				| Squeeze the cube into one dimension			| (?,8,8,480)           |    (?,30720) |
| Fully connected		| scope:fully_2; pairwise connections between all nodes	    | (?,30720)       | (?,200)         |
| Fully connected		|  scope:fully_3; pairwise connections between all nodes		        | (?,200)       | (?,400)         |
| Fully connected		|  scope:fully_4; pairwise connections between all nodes	  		| (?,400)           | (?,300)             |
| Fully connected		|  scope=logits; pairwise connections between all nodes	  		| (?,300)          | (?,43)            |


#### Visualization of the model 
![Model architecture](/documentation/model.png)
### Model Training

#### Hyperparameter tuning  
* LEARNING RATE = 0.0005
* EPOCHS = 350 
* BATCH SIZE = 128
* Dropout keep probability rate : 0.5

#### Optimizer 
Adam optimizer

### Solution Approach

#### Description:
There was overfitting at the beginning of the training, so I changed the number of inputs/outputs of the layers.

Overfitting reduction steps :
* Modifying/Changing the size of layers, and playing with the architecture, helped a lot :)
* Tried data augmentation and data replication (Didn't help)
* Added dropout regularization and achieved improvements,  but also tried dropout connect, batch normalization and l2 (Didn't help).

After my final model was constructed, It took me about ***1 hour*** and ***30 minutes*** to train on GeForce 1080Ti on 350 iterations.
After 1 hour and 30 minutes, you should get 99.1%-99.4% accuracy on the train set with ***learning_rate=0.0005***,
Then you can start decreasing the learning rate and you can easily achieve 99.45% on the validation set.
On every new loss/performance minimum, I was decreasing the learning rate, and it's final value is 0.00001


#### The final results are:
* Train Accuracy = 100%
* Validation Accuracy = 99.47845804988662%
* Test Accuracy = 98.1472684010002%

## Test a Model on New Images

### Acquiring New Images

#### Images
The submission includes 8 new German Traffic signs found on the web (see: [utils.py/download_files]](/utils.py).

![Download data](/documentation/download_data.png)

#### Discussion
The new images are chosen carefully, they are similar to the one we trained, but with a noise around : 
* Sky behind the sign
* Snow behind the sign
* Nature behind the sign
* Sign within Pane/Platform 
* Snow and Camera on the sign
* Car/road behind the sign
I think that this is the most common noise that we will see in real life and I challenged the neural network to see how it will perform.

### Performance on New Images
Bellow, there is info about the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

#### Visualization 
![Labeled data](/documentation/labeled_data.png)

#### Performance : 
* Accuracy 100.0
* Loss 0.13730292

Compared to the validation, we had : 
* loss:  0.06094360592724713
* Accuracy 99.45
Which is not pretty bad, in comparison.

### Model Certainty - Softmax Probabilities
The top five softmax probabilities of the predictions on the captured images are outputted. 

#### Visualization 
![Probabilities data](/documentation/probabilities_data.png)

#### Discussion
We can clearly see that the softmax probabilities are perfectly indicating the class label with almost 100% on all the signs, except the ***Beware of ice/snow*** where the indication is very very close ***Speed limit 80km/h***
So, generally, this means that there are some features on this picture that was not trained by our model, decreasing the indication probability.

