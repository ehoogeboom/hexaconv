
from chainer import Chain
import chainer.functions as F


class ConvBNAct(Chain):

    def __init__(self,
                 conv,
                 bn=True,
                 act=F.relu):
        super(ConvBNAct, self).__init__(conv=conv)

        if bn:
            out_channels = self.conv.W.data.shape[0]
            self.add_link('bn', F.BatchNormalization(out_channels))
        else:
            self.bn = None

        self.act = act

    def __call__(self, x, train, finetune):
        import cupy as cp

        # assert cp.isfinite(x.data).all()
        y = self.conv(x)
        # assert cp.isfinite(y.data).all()

        if self.bn:
            y = self.bn(y, test=not train, finetune=finetune)

            # assert cp.isfinite(y.data).all()
        if self.act:
            y = self.act(y)
            # assert cp.isfinite(y.data).all()

        # print x.data.shape, '->', y.data.shape

        return y
