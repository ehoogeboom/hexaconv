
# Filter masks provide a simple way to reduce the number of parameters per filter, for filters on groups.
# Just like planar filters that are only supported on a small region (e.g. 3x3),
# filters on groups can have a limited support.
# For split groups that consists of a stabilizer that fixes the origin H, and a translation group T,
# a filter masks has a 0 for transformations in H where the filter should have value 0,
# and 1 for transformations in H where the filter should not be zero.

# A downside of the current implementation of G-convs with filter masks is that we still compute convolutions
# for 2D filter planes that are zero. This could be fixed by incorporating masks in the 2D convolution routines.

import numpy as np


def make_p4m_masks_segregate_mirrors(out_channels, in_channels):
    """
    Construct a mask for a set of p4m filters that limits their support to non-flipped transformations.

    Since the p4m-conv will translate, rotate and *flip* these filters, we will nevertheless filter both orientations.
    A p4m-conv that uses the mask produced by this function will process two "mirrored information streams"
    separately. If we stack a number of such layers, it is as if we apply a p4-conv to the input and the flipped input
    (they don't interact).
    """
    mask = np.zeros((out_channels, in_channels, 8))
    mask[:, :, :4] = 1.
    return mask


def make_p4m_masks_e_and_m_filters(out_channels, in_channels, num_e_filters):
    assert num_e_filters <= out_channels
    mask = np.zeros((out_channels, in_channels, 8))
    mask[:num_e_filters, :, :4] = 1.
    mask[num_e_filters:, :, 4:] = 1.
    return mask


def make_p4m_masks_e_and_m_channels(out_channels, in_channels, num_e_channels):
    assert num_e_channels <= in_channels
    mask = np.zeros((out_channels, in_channels, 8))
    mask[:, :num_e_channels, :4] = 1.
    mask[:, num_e_channels:, 4:] = 1.
    return mask


# random
