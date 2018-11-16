# Crack Attack
The purpose of this project was to train a neural net to detect if a picture of concrete contained a crack or not.

### Data
The data was sourced from 'Link' and consisted of labeled crack and no crack data.  The images are 256 X 256 with RGB color.  The images are very noisy and many of the labeled crack images did not contain cracks.  


### Model - Neural Nets
For this project I decided to try and implement a neural net. A neural net is a deep learning tool that adds hidden layers into  model.  These layers add weights to the model that humans can have a tough time understanding.  


### Initial Neural Net
The initial net I decided to implement was based off the Keras image classification blog, https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html.  The net consists of three convolutional layers with relu activation functions, a dense layer with relu activation and 0.5 drop out and a final dense layer with a sigmoid activation function.

#### Model Performance
On paper this model looked great during training.  Accuracies ranged from 80-90%.  The issue was that the losses were high 1.5-1.8.  Upon digging into this I saw that I was only testing on no crack images.  This means I was getting only one guess on the test and train sets giving a high accuracy and high loss.

Once the test and train data generators were fixed for sending both classes of images in the model got 50% accuracy with lower losses of roughly 0.7.  This model made much more sense.  

### Back to the drawing board
The next model I built added complexity with two extra hidden layers and increased the number of filters at each layer of the model.  

#### Model Performance
This model performed some what better with accuracies of around 65% and losses at 0.6 with only 25 epochs.  This model performed much better and showed promise with the lower loss values.  

Lets look into the convolutional layers to see if we can see anything that the model is picking up on.

First convolution layer

<img src="images/stitched_filters_6x6_second_nn_conv2d_1.png" height="480" width="480">


Second convolution layer

<img src="images/stitched_filters_5x5_second_nn_conv2d_2.png" height="480" width="480">




Third convolution layer

<img src="images/stitched_filters_4x4_second_nn_conv2d_4.png" height="480" width="480">


Forth convolution layer

<img src="images/stitched_filters_2x2_second_nn_conv2d_5.png" height="480" width="480">




##### What is the model misclassifying
The images are very 'noisy' they contain mostly dead image space with sparse cracks.  Looking though the images it appears that most edge cases get missed labeled.  It is also tough for the model to detect cracks when the concrete has aggregate base.  
![]()
![]()
![]()
![]()
![]()
![]()

##### GPU Issue!!


![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)
