# Digit Recognition Tree
### Team Members: Twee Shackelford, Richard M Sullivan. Group 6.
### Link to the competition:
https://www.kaggle.com/competitions/digit-recognizer/overview 
### Description the problem: 
Given a set of images of handwritten numbers, discern which numbers are depicted in the images.
### The kind of data will be used and its size:
The MNIST dataset from 1999 is a sample of handwriting, from which images depicting written numbers have been extracted and converted into a CSV file with features for every pixel in the file which contains the value of that pixel. There are 1571 features in this dataset representing each pixel of the images.
### The kind of machine learning task:
The machine learning task is to classify input images as one of ten digits classes, 0 - 9, thus the problem is one of classification.
### Brief description of the kind of machine learning methods that are anticipated to be used:
It is almost certain that some of the 1571 total features in the dataset will contain some features that contain almost no useful information, and thus we will most likely have to employ some form of Feature Reduction to reduce the dimensionality. To this end RFE will be used for its efficiency and simplicity of implementation.
To improve model selection we will also employ k-fold cross validation on a decision tree model, as it is both simple to implement and efficient at multi-class classification problems.


