# CNN for sentiment analysis

## Overview
This project has been created from my own goal of implementing a research paper for the first time. Yoon Kim's [*CNN for Sentence Classification*](https://arxiv.org/abs/1408.5882) is something that I have some prior domain experience with my attempt to build an LSTM recurrent neural network (found [here](https://gitlab.com/bigbawsboy/IMDB-sentiment-analysis)).

My approach in this notebook is to go through the different models that the paper has created and performed ablation studies on various datasets.

Lastly, this project's other goal is to dive in deeper with the PyTorch framework and writing more streamlined functions to avoid code repeatability.

## Model architectures
In the paper, there are four various CNN architectures created.

### 1) `CNN-rand`
In this model, the architecture goes as follows: `Embedding` -> `Conv2d` -> `MaxPool1d` -> `Dropout` -> `Linear`. The `Embedding` layer uses no pre-trained word embeddings and its parameters are learned during training.

### 2) `CNN-static`
In this model, it has the same architecture as in `CNN-rand`. The only difference is the `Embedding` layer. A pre-trained word embedding, **GloVe**, is used and its parameters are not learned during training (i.e., setting `requires_grad` to `False` in this layer).

### 3) `CNN-non-static`
In this model, it has the same architecture as `CNN-rand`. Unlike `CNN-static`, this model enables the learning of the `Embedding` layer's parameters. Lastly, a pre-trained word embedding is used.

### 4) `CNN-multichannel`
In this model, it combines the idea of having a "2-channel" embedding matrix: static and non-static channels. Note that it has the same architecture as all of the other models above. The idea of this model is to learn various contexts for the embedding layers. The static channel is used to regulate the learned parameters in the non-static channel. As a result, the word embeddings of the static channel still maintains the relationship of words from the pre-trained word embedding. Meanwhile, the non-static channel gets to learn more about the relationship of words with others based on the movie reviews context.

## Credits
Thank you to Yoon Kim's [paper](https://arxiv.org/abs/1408.5882) for inspiring me with creating a CNN-based sentiment analysis model. Also, thanks to [this repository](https://github.com/bentrevett/pytorch-sentiment-analysis) (in particular, Juypter notebook titled **4 - Convolutional Sentiment Analysis**) for guiding me into starting the scaffolding for the `CNN-rand` model architecture.
