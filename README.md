# DeepLearning_JSCC

This project is part of the Deep Learning course from DTU, it belongs to:

- Angel Daniel Cester Sala, [s222882@dtu.dk](mailto:s222882@dtu.dk)
- Vicente Bosch Alonso, [s222928@dtu.dk](mailto:s222928@dtu.dk)
- Enrique Jose Aguado Andres, [s222930@dtu.dk](mailto:s222930@dtu.dk)
- Victor Luna Santos, [s222931@dtu.dk](mailto:s222931@dtu.dk)

## Introduction
When sending an image from one device to another we encode the image on the transmitter, send the data through a noisy channel, and decode it on the receiver side. This encoding and decoding are traditionally done by mathematical-based algorithms. A novel that got a lot of attention in this area is described in previous works [[1]](https://arxiv.org/abs/1809.01733) [[2]](https://arxiv.org/abs/1903.06333) [[3]](https://arxiv.org/abs/1911.11174), where a simple autoencoder is used to do the encoding and decoding. The images that the autoencoder trained on are of size 32x32 pixels, and the performance drops down as we increase the size. One approach is to partition a larger image into blocks of 32x32 and send them one by one and at the end, put them back together. The idea beyond this project is to optimize this process, selecting which model is needed to use to have the best image reconstructed with the minimal of variables sent. Thereafter, we can send the parts containing more information with more data and the parts having less information with less data to get a better overall performance total.

## How to run it
All the code for this project have been runned into Google Colab, so it is possible that some adjustment is needed in case to be runned locally.

First of all we need to install the needed libraries, for it, a virtual environment is recommended. To install it, you can simply run:
```
pip install -r requirements.txt
```

Once it has been done, you will need to download the dataset, you can download imagenet-a from [here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar) and imagenetv2 from [here](https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz?download=true), and place it in their respective folders inside [data/raw/](data/raw/).

Finally, you can run the notebooks in [notebooks](notebooks/), you will find [Training_JSCC.ipynb](notebooks/Training_JSCC.ipynb) where you can retrain the models and [Evaluate_JSCC.ipynb](notebooks/Evaluate_JSCC.ipynb) where you can evaluate with the new metrics which model is performing better.

## Acknowledgements
The authors wish to thank Amin Hasanpour of DTU for supervising and helping with the project.