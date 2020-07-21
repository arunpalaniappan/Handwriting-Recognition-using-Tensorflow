# Handwritten Text Recognition using Tensorflow

The work here is an implementation of Stage 3 Dual stream architecture in [Fully Convolutional Networks for Handwriting Recognition](https://arxiv.org/abs/1907.04888). 

The neural network takes in handwriting sample of unknown length and outputs an arbitary sequence of characters. We achieved an accuracy of 34% and further improvements have to be made.

## Dataset

The dataset used here is the IAM dataset available at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). Then follow these instructions:
1. Create a directory `Data` inside this directory.
2. Download `ascii/words.txt`  and place the files `words.txt` in `Data` directory.
3. Download `words/words.tgz` and place the contents of the extracted file in 'Data/words' directory.

## Model Overview:
The dual stream architecture consists of three neural networks. Two streams are convolutional neural network rchitectures which are used to capture the local context and global context.
The output of the two streams are combined and passed to the third stream which classifies the outputs of neural networks into symbols. We have used infinity-norm as loss function. Cross-entropy loss can also be used as loss function and it is to be tested.


## Other References:

\[1\] [SimpleHTR](https://github.com/githubharald/SimpleHTR)

Disclaimer: We are in no way related to the authors of the paper.
