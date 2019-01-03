# Tensorflow Tutorials

This repo holds a reference to my progress through the Tensorflow [Tutorial](https://www.tensorflow.org/tutorials/).
I digest material best through typing out the code myself and researching snippets I don't understand.  

## Basic Classification

The first basic classification tutorial runs through a simple image recogintion model.  Through which I learned about
preprocessing the training data in the same way as the test data.  The preprocessing consisted of scaling the pixel values
into a range from 0 to 1, inclusive. To accomplish this, each byte was dividied by 255 then cast to a float value.

I came to understand the basic ideas of layers in a neural network, i.e., the building blocks that extract meaningful 
representations from the data.  For the basic classification model, the layers consisted of flattening the 2-D array 
of pixels into a 1-D array.  Then a 128-node (or neuron) densely connected layer was added.  The last layer was a 10-node
*softmax* layer.  The softmax layer output an array of 10 probability scores that sum to 1.  Each node contains the score
of a particular image belonging to a particular class.
