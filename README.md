# HexaConv

## Abstract
The effectiveness of Convolutional Neural Networks stems in large part from their ability to exploit the translation invariance that is inherent in many learning problems. Recently, it was shown that CNNs can exploit other invariances, such as rotation invariance, by using group convolutions instead of planar convolutions. However, for reasons of performance and ease of implementation, it has been necessary to limit the group convolution to transformations that can be applied to the filters without interpolation. Thus, for images with square pixels, only integer translations, rotations by multiples of 90 degrees, and reflections are admissible.

Whereas the square tiling provides a 4-fold rotational symmetry, a hexagonal tiling of the plane has a 6-fold rotational symmetry. In this paper we show how one can efficiently implement planar convolution and group convolution over hexagonal lattices, by re-using existing highly optimized convolution routines. We find that, due to the reduced anisotropy of hexagonal filters, planar HexaConv provides better accuracy than planar convolution with square filters, given a fixed parameter budget. Furthermore, we find that the increased degree of symmetry of the hexagonal grid increases the effectiveness of group convolutions, by allowing for more parameter sharing. We show that our method significantly outperforms conventional CNNs on the AID aerial scene classification dataset, even outperforming ImageNet pre-trained models.

The full paper is available on [openreview](https://openreview.net/forum?id=r1vuQG-CW).

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
