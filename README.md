# Comp4901J Deep Learning In Computer Vision - Assignment3

The course website: https://course.cse.ust.hk/comp4901j/

Image Captioning, Network Visualization, Style Transfer, Generative Adversarial Networks

In this assignment you will implement recurrent networks, and apply them to image captioning on Microsoft COCO. You will also explore methods for visualizing the features of a pretrained model on ImageNet, and also this model to implement Style Transfer. Finally, you will train a generative adversarial network to generate images that look like a training dataset!

Note from CK: please complete only the Tensorflow notebook for HKUST version of the assignmenet because our WinPython package only supports Tensorflow.
The goals of this assignment are as follows:

Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time
Understand and implement both Vanilla RNNs and Long-Short Term Memory (LSTM) RNNs
Understand how to sample from an RNN language model at test-time
Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system
Understand how a trained convolutional network can be used to compute gradients with respect to the input image
Implement and different applications of image gradients, including saliency maps, fooling images, class visualizations.
Understand and implement style transfer.
Understand how to train and implement a generative adversarial network (GAN) to produce images that look like a dataset.

Download data:
Once you have the starter code, you will need to download the COCO captioning data, pretrained SqueezeNet model (TensorFlow-only), and a few ImageNet validation images. Run the following from the assignment3 directory:

cd cs231n/datasets folder and click get_assignment3_data.py to run the Python code to download the data.
Submitting your work:
Whether you work on the assignment locally or in the labs, once you are done working run the Python code collectSubmission.py; this will produce a file called assignment3.7z. Upload this file under CASS. You can find CASS instruction of uploading your submission here.

You can do Questions 3, 4, and 5 in TensorFlow or PyTorch. There are two versions of each notebook, with suffixes -TensorFlow or -PyTorch. No extra credit will be awarded if you do a question in both TensorFlow and PyTorch.
Q1: Image Captioning with Vanilla RNNs (25 points)
The Jupyter notebook RNN_Captioning.ipynb will walk you through the implementation of an image captioning system on MS-COCO using vanilla recurrent networks.

Q2: Image Captioning with LSTMs (30 points)
The Jupyter notebook LSTM_Captioning.ipynb will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images (15 points)
The Jupyter notebooks NetworkVisualization-TensorFlow.ipynb /NetworkVisualization-PyTorch.ipynb will introduce the pretrained SqueezeNet model, compute gradients with respect to images, and use them to produce saliency maps and fooling images. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awardeded if you complete both notebooks.

Q4: Style Transfer (15 points)
In the Jupyter notebooks StyleTransfer-TensorFlow.ipynb/StyleTransfer-PyTorch.ipynb you will learn how to create images with the content of one image but the style of another. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awardeded if you complete both notebooks.

Q5: Generative Adversarial Networks (15 points)
In the Jupyter notebooks GANs-TensorFlow.ipynb/GANs-PyTorch.ipynb you will learn how to generate images that match a training dataset, and use these models to improve classifier performance when training on a large amount of unlabeled data and a small amount of labeled data. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awarded if you complete both notebooks.

