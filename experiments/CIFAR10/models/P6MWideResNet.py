
from math import sqrt

from chainer import ChainList
import chainer.links as L
import chainer.functions as F
from experiments.wide_residual_block_2d import WideResBlock2D
from groupy.gconv.gconv_chainer.p6m_conv_axial import P6MConvZ2, P6MConvP6M


class P6MWideResNet(ChainList):

    def __init__(self, num_blocks=4, nc32=5, nc16=10, nc8=20, k=1,
                 drop_ratio=0.0):
        """
        :param num_blocks: the number of resnet blocks per stage. There are 3
            stages, for feature map width 32, 16, 8.
        Total number of layers is 6 * num_blocks + 2
        :param nc32: the number of feature maps in the first stage (where
            feature maps are 32x32)
        :param nc16: the number of feature maps in the second stage (where
            feature maps are 16x16)
        :param nc8: the number of feature maps in the third stage (where
            feature maps are 8x8)
        """
        ws = sqrt(2.)  # This makes the initialization equal to He et al.

        super(P6MWideResNet, self).__init__()

        # The first layer is always a convolution.
        self.add_link(
            P6MConvZ2(in_channels=3, out_channels=nc32, ksize=3, stride=1,
                      pad=1, wscale=ws)
        )
        # Add num_blocks ResBlocks (2n layers) for the size 32x32 feature maps
        for i in range(num_blocks):
            nc_in = nc32 * k if i > 0 else nc32
            self.add_link(
                WideResBlock2D(
                    in_channels=nc_in, out_channels=nc32 * k, wscale=ws,
                    downsample=False, ConvLink=P6MConvP6M,
                    drop_ratio=drop_ratio))

        # Add num_blocks ResBlocks (2n layers) for the size 16x16 feature maps
        # The first convolution uses stride 2
        for i in range(num_blocks):
            nc_in = nc16 * k if i > 0 else nc32 * k
            downsample = i == 0
            self.add_link(
                WideResBlock2D(
                    in_channels=nc_in, out_channels=nc16 * k, wscale=ws,
                    downsample=downsample, ConvLink=P6MConvP6M,
                    drop_ratio=drop_ratio))

        # Add num_blocks ResBlocks (2n layers) for the size 8x8 feature maps
        for i in range(num_blocks):
            nc_in = nc8 * k if i > 0 else nc16 * k
            downsample = i == 0
            self.add_link(
                WideResBlock2D(
                    in_channels=nc_in, out_channels=nc8 * k, wscale=ws,
                    downsample=downsample, ConvLink=P6MConvP6M,
                    drop_ratio=drop_ratio))

        # Add BN and final layer
        # We do ReLU and average pooling between BN and final layer, but these
        # don't require a link.
        self.add_link(F.BatchNormalization(size=nc8 * k))
        self.add_link(
            L.Convolution2D(in_channels=nc8 * k * 12, out_channels=10, ksize=1,
                            wscale=ws))

    def __call__(self, x, t, train=True, finetune=False):
        h = x

        # First conv layer
        h = self[0](h)

        # Residual blocks
        for i in range(1, len(self) - 2):
            h = self[i](h, train, finetune)

        # BN, relu, pool, final layer
        h = self[-2](h)
        h = F.relu(h)
        n, nc, ns, nx, ny = h.data.shape
        h = F.reshape(h, (n, nc * ns, nx, ny))
        h = F.average_pooling_2d(h, ksize=h.data.shape[2:])
        h = self[-1](h)
        h = F.reshape(h, h.data.shape[:2])

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def start_finetuning(self):
        for c in self.children():
            if isinstance(c, WideResBlock2D):
                c.bn1.start_finetuning()
                c.bn2.start_finetuning()

    def number_of_params(self):
        total = 0
        for c in self.children():
            if isinstance(c, WideResBlock2D):
                for c_ in c.children():
                    if isinstance(c_, P6MConvP6M):
                        total += 7./9 * sum([p.data.size for p in c_.params()])
                    else:
                        total += sum([p.data.size for p in c_.params()])
            elif isinstance(c, P6MConvZ2):
                total += 7./9 * sum([p.data.size for p in c.params()])
            elif isinstance(c, F.BatchNormalization):
                total += sum([p.data.size for p in c.params()])
            elif isinstance(c, F.Convolution2D):
                total += sum([p.data.size for p in c.params()])
            else:
                raise ValueError('Unknown instance')

        return total
