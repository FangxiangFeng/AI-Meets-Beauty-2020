# AI-Meets-Beauty-2020

This repository contains the implementation for paper[ *Learning Visual Features from Product Title for Image Retrieval*]. The paper introduces a method to retrieve product images by image query. The key idea of our method is to convert the product title to discrete labels as the semantic supervised signals of image feature learning. This method achieves the fourth position in the Grand Challenge of AI Meets Beauty in 2020 ACM Multimedia by using only a single ResNet-50 model without any human annotations and pre-processing or post-processing tricks.

# Convert the product title to discrete labels

See the code in `generate_labels.py`. 

# Evaluation

### Download pretrained ResNet-50 model

You can grab our pretrained ResNet-50 model from [here](https://drive.google.com/file/d/1BH9yn7daO4V3fPQ142sREQVh3rT2CllE/view?usp=sharing) (~226MB). Put the downloaded file `resnet50bt.pth.tar` in the directory "pretrained". This model is trained by [BiT toolkit](https://github.com/google-research/big_transfer).

### Download the features of the candidates images

You can grab the features of the candidates images from [here](https://drive.google.com/file/d/1tyZEryLWJ_fG4e-XJgDaMDYEi1zf35X6/view?usp=sharing) (~2.1GB). Put the downloaded file `feat_resnet50bt_mac.pkl` in the directory "feature".

### Run the script `predict`



## Acknowledges

Some of the code is adapted from [here] (https://github.com/gniknoil/Perfect500K-Beauty-and-Personal-Care-Products-Retrieval-Challenge).
