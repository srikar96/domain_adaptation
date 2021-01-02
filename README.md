# Unsupervised Domain Adaptation for Synthetic to Real Images
## Some of the challenges faced in Deep Learning
*	Lack of enough data
*	Difficulty in obtaining good quality data
*	Impractically in Annotating the dataset
*	Imbalances in the dataset

## Unsupervised Domain Adaptation
#### What is it?
* Unsupervised Domain Adaptation aims to classify unlabeled target domain by transferring knowledge from labeled source domain with domain shift.
* The network is trained to learn the features of both the source and target domain such that the learnt features are domain independant. 

#### Advantages
Synthetic data can be easily generated using computers (CAD softwares). This synthetic - labelled data can be used to train Domain Adaptation models to learn the target distributation from the source.

## Dataset 
* DomainNet dataset was used to train the model.
* The dataset is made up of images of everyday objects. It consists of 6 different domains (Sketch, Real World, 2D drawing etc). 
* For this project, two domains were used (Real Images and Sketch Images) which act as source (10 Classes) and target (10 Classes). 
* Real Images count: ~120,000
* Sketch Images count: ~69000
Link to the dataset: http://ai.bu.edu/M3SDA/

## Architecture
* RevGrad (Reverse Gradient) architecture was used for the model.
* The architecture is based on Generative Adversal Networks (GANs) which has a Generator and a Discriminator.
* The network can be broken down into 3 parts. 
	* The first part is the common path which is used to extract features from the images.
	* The second part is the label predictor which is similar to a simple image classifier network which learns the class labels.
	* The third part of the network is the domain classifier which is a binary classifier used to predict the domain the image belongs.
* The goal is to minimize the classification loss and maximize the domain loss. This is achieved by using a gradeint reversal layer to change the sign of the gradinets while backpropagation.
* The source images are passed onto both the paths while training whereas the target domain images are only passed through the domain classifier.
* This way the network learns the features that are domain independant.

## Results
| **Architecture** | **Configuration** | **Target Accuracy** |
|--|--|--|
| Baseline | ResNet-50 | 64.08% |
| ResNet-50 | training the last 2 layers + mutiple linear layers| 72.13% | 
|  | training the last 2 layers + single softmax| 74.11% |
|  | training the last layer + single softmax | 78.93% |
| ResNet-101 | training the last layer + single softmax | 79.47% |
| ResNet-152 | training the last layer + single softmax | 79.57% |
| GooleNet | training the last 2 layers + mutiple linear layers| 67.96% |
|  | training the last layer + single softmax | 72.40% |

## Folder Contents
**Baseline**: Contains the code for the baseline model

**RevGrad**: Contains the code for the RevGrad Architecture



 


