# ML notebooks

## Deep Learning

- All the notebooks use PyTorch to as the DL framework.
- [Tips for training DL models](https://basujindal.me/python/deep%20learning/machine%20learning/computer%20vision/2021/06/15/tips-for-deep-learning.html)

## Computer Vision

- [conv_example.py](CV/conv_example.py): Python code to train a Convolution NN on custom dataset, using pytorch dataloader. (End2end example)

- [GAN.ipynb](CV/conv_example.ipynb): Train a DCGAN on CelebA dataset to generate fake faces. 

- [DCGAN.ipynb](CV/DCGAN.ipynb): Train a DCGAN on MNIST dataset to generate fake digits.

## GAN

- Training challenges:
-- Use small batch size especially if model is small, large batch size may train Discriminator very fast during early phase of training.    


### NLP

- [dataprep_bpe.ipynb](NLP/dataprep_bpe.ipynb): Example notebook to load custom Language dataset for Machine translation and tokenize it using Transformers Library with Byte Pair Encoding.

- [transformers.ipynb](NLP/transformers.ipynb): Train a custom transformer model for Machine Translation

### RL

- [reinforce_cartpole.ipynb](RL/reinforce_cartpole.ipynb): REINFORCE algorithm from scratch to solve a simple cartpole envionment. Notebook taken from [Yandex RL course](https://github.com/yandexdataschool/Practical_RL)
