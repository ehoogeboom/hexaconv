# HexaConv

## Abstract
The effectiveness of Convolutional Neural Networks stems in large part from their ability to exploit the translation invariance that is inherent in many learning problems. Recently, it was shown that CNNs can exploit other invariances, such as rotation invariance, by using group convolutions instead of planar convolutions. However, for reasons of performance and ease of implementation, it has been necessary to limit the group convolution to transformations that can be applied to the filters without interpolation. Thus, for images with square pixels, only integer translations, rotations by multiples of 90 degrees, and reflections are admissible.

Whereas the square tiling provides a 4-fold rotational symmetry, a hexagonal tiling of the plane has a 6-fold rotational symmetry. In this paper we show how one can efficiently implement planar convolution and group convolution over hexagonal lattices, by re-using existing highly optimized convolution routines. We find that, due to the reduced anisotropy of hexagonal filters, planar HexaConv provides better accuracy than planar convolution with square filters, given a fixed parameter budget. Furthermore, we find that the increased degree of symmetry of the hexagonal grid increases the effectiveness of group convolutions, by allowing for more parameter sharing. We show that our method significantly outperforms conventional CNNs on the AID aerial scene classification dataset, even outperforming ImageNet pre-trained models.

The full paper is available on [openreview](https://openreview.net/forum?id=r1vuQG-CW).

## Running the experiments
This will run the cifar experiment on the P4WideResNet.py network:

    python train_cifar.py --modelfn=experiments/CIFAR10/models/P4WideResNet.py --epoch 300 --save_freq=100 --gpu 0 --opt=MomentumSGD --lr_decay_factor=0.1 --lr_decay_schedule=50-100-150 --batchsize 125 --transformations='' --opt_kwargs="{'lr':0.05}" --datadir=/path/to/cifar10 --resultdir=/path/to/results
    

This will run the cifar experiment on the P6WideResNet.py network. Note the addition of the hex sampling flag.

    python train_cifar.py --modelfn=experiments/CIFAR10/models/P6WideResNet.py --epoch 300 --save_freq=100 --gpu 0 --opt=MomentumSGD --lr_decay_factor=0.1 --lr_decay_schedule=50-100-150 --batchsize 125 --transformations='' --opt_kwargs="{'lr':0.05}" --datadir=/path/to/cifar10 --resultdir=/path/to/results --hex_sampling axial
    
This will run the AID experiment on the P4WideResNet.py network:

    python train_aid.py --modelfn=experiments/AID/models/P4WideResNet.py --epoch 300 --save_freq=100 --gpu 0 --opt=MomentumSGD --lr_decay_factor=0.1 --lr_decay_schedule=50-100-150 --batchsize 100 --transformations='' --opt_kwargs="{'lr':0.05}" --datadir=/path/to/aid --resultdir=/path/to/results
    

This will run the AID experiment on the P6WideResNet.py network. Note the addition of the hex sampling flag.

    python train_aid.py --modelfn=experiments/AID/models/P6WideResNet.py --epoch 300 --save_freq=100 --gpu 0 --opt=MomentumSGD --lr_decay_factor=0.1 --lr_decay_schedule=50-100-150 --batchsize 100 --transformations='' --opt_kwargs="{'lr':0.05}" --datadir=/path/to/aid --resultdir=/path/to/results --hex_sampling axial

Performing experiments with other networks only requires a chance to the modelfn flag. Data for the experiments is available here [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) and here [AID](http://www.lmars.whu.edu.cn/xia/AID-project.html).

## General Usage
The following template shows how to use p6 convolutions
```python
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable
from groupy.hexa.hexa_sample import sample_cartesian2axial

from groupy.gconv.gconv_chainer.p6_conv_axial import P6ConvP6, P6ConvZ2


class Network(chainer.ChainList):
    def __init__(self, n_channels=16, n_outputs=10):
        super(Network, self).__init__()

        ws = np.sqrt(2.)
        self.invariant = False  # Forces the final layer to be invariant

        # First conv layer
        self.add_link(
            P6ConvZ2(in_channels=3, out_channels=n_channels,
                     ksize=3, stride=1, pad=1, wscale=ws)
        )

        # Second conv layer.
        self.add_link(
            P6ConvP6(in_channels=n_channels, out_channels=n_channels,
                     ksize=3, stride=1, pad=1, wscale=ws)
        )

        # Fully connected layer
        in_channels = n_channels if self.invariant else n_channels * 6
        self.add_link(
            L.Convolution2D(in_channels=in_channels,
                            out_channels=n_outputs, ksize=1, wscale=ws))

    def __call__(self, x, t):
        # P6ConvZ2 convolution
        h = self[0](x)

        # P6ConvP6 convolution
        h = self[1](h)

        # Dimensions of h
        n, nc, ns, nx, ny = h.data.shape

        if self.invariant:
            # Invariance, recommended when orientations do not influence label
            # function (e.g. AID). Pooling over p6.
            h = F.sum(h, axis=-1)
            h = F.sum(h, axis=-1)
            h = F.sum(h, axis=-1)
            h /= ns * nx * ny
            h = F.reshape(h, (n, nc, 1, 1))

        else:
            # Equivariance, recommended when orientations may influence label
            # function (e.g. CIFAR-10). Pooling over Z2.
            h = F.reshape(h, (n, nc * ns, nx, ny))
            h = F.average_pooling_2d(h, ksize=(nx, ny))

        # Fully connected layer.
        h = self[-1](h)
        h = F.reshape(h, h.data.shape[:2])

        # Return cross-entropy and accuracy.
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)


gpu = 0

# Get model.
model = Network().to_gpu(gpu)

# Generate image data with labels
X = np.random.randn(500, 3, 32, 32).astype('float32')
y = np.random.randint(10, size=500).astype('int32')

# Sample to hex grid
X_hex = sample_cartesian2axial(X)

# Get batches on gpu.
x_batch = Variable(cuda.to_gpu(X_hex, device=gpu))
y_batch = Variable(cuda.to_gpu(y, device=gpu))

# Forward pass.
loss, acc = model(x_batch, y_batch)

```

## Citation
To cite HexaConv in publications use

    Emiel Hoogeboom, Jorn W.T. Peters, Taco S. Cohen, and Max Welling. HexaConv. International Conference on Learning Representations, 2018.

A BibTeX entry for LaTeX users is

```
@inproceedings{
hoogeboom2018hexaconv,
title={HexaConv},
author={Emiel Hoogeboom and Jorn W.T. Peters and Taco S. Cohen and Max Welling},
booktitle={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1vuQG-CW},
}
```
