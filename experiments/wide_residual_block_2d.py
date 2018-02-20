import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np


# New style residual block
class WideResBlock2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, wscale=np.sqrt(2), bias=0,
                 drop_ratio=0.3, downsample=False, ConvLink=None):
        ksize = 3
        pad = 1
        stride = 1 if not downsample else 2
        fiber_map = 'id' if in_channels == out_channels else 'linear'
        self.drop_ratio = drop_ratio

        assert ConvLink is not None, "Specify convolution"
        assert ksize % 2 == 1

        if not pad == (ksize - 1) // 2:
            raise NotImplementedError()

        bn1 = L.BatchNormalization(in_channels)
        conv1 = ConvLink(
            in_channels=in_channels, out_channels=out_channels, ksize=ksize,
            stride=stride, pad=1, wscale=wscale)
        bn2 = L.BatchNormalization(out_channels)
        conv2 = ConvLink(
            in_channels=out_channels, out_channels=out_channels, ksize=ksize,
            stride=1, pad=1, wscale=wscale)

        super(WideResBlock2D, self).__init__(
            bn1=bn1, conv1=conv1, bn2=bn2, conv2=conv2)

        if fiber_map == 'id':
            self.fiber_map = F.identity
        elif fiber_map == 'linear':
            fiber_map = ConvLink(
                in_channels=in_channels, out_channels=out_channels, ksize=1,
                stride=stride, pad=0, wscale=wscale)
            self.add_link('fiber_map', fiber_map)

    def __call__(self, x, train, finetune):

        h = self.conv1(F.relu(self.bn1(x, test=not train, finetune=finetune)))
        if self.drop_ratio > 0.0:
            h = F.dropout(h, ratio=self.drop_ratio, train=train)
        h = self.conv2(F.relu(self.bn2(h, test=not train, finetune=finetune)))

        # Apply the fiber map
        hx = self.fiber_map(x)

        return hx + h
