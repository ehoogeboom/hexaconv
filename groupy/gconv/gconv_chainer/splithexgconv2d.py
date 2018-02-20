from groupy import hexa
import splitgconv2d


class SplitHexGConv2D(splitgconv2d.SplitGConv2D):
    """
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super(SplitHexGConv2D, self).__init__(*args, **kwargs)

        filter_mask = hexa.mask.hexagon_axial(self.ksize)
        filter_mask = filter_mask[None, None, None, ...].astype(self.dtype)
        self.add_persistent('filter_mask', filter_mask)

    def __call__(self, x):
        # Apply a mask to the parameters
        self.W.data = self.W.data * self.filter_mask

        y = super(SplitHexGConv2D, self).__call__(x)

        # Get a square shaped mask if it does not yet exist.
        if not hasattr(self, 'output_mask'):
            ny, nx = y.data.shape[-2:]
            self.add_persistent(
                'output_mask',
                self.xp.array(
                    hexa.mask.square_axial(ny, nx)[None, None, None, ...]))

        y = y * self.output_mask

        return y
