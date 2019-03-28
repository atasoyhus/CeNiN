# CeNiN
**C**e**N**i**N** (means "fetus" in Turkish) is an implementation of feed-forward phase of deep **C**onvolutional **N**eural **N**etworks in pure C#. It doesn't require any external library and can be used in all programming languages supported by .NET.

![CeNiN screenshot](screenshot.png)

## Pretrained Models
You can download two pretrained CeNiN models. These are actually two of VGG16 models of Oxford Visual Geometry Group. Parameters of those models are stored in ".cenin" files that allow millions of parameters to be loaded into memory quickly under .NET framework.  
- [imagenet-matconvnet-vgg-f.cenin (19 layers)](models/imagenet-matconvnet-vgg-f.cenin)
- [imagenet-vgg-verydeep-16.cenin (37 layers)](models/imagenet-vgg-verydeep-16.cenin)

## Performance
The most time-consuming layers are convolution layers. The other layers are fast enough without extra optimization.  
***Conv_1*:** It takes more than 1 minute to pass an image through all the layers.  
***Conv_2*:** I used the approach explained in a paper by Nvidia (no, not using GPUs :D). It is not much faster than the first one because of the indexing approach that I used in *Tensor* class.  
***Conv_3*:** The same as *Conv_2*, but this time with non-repetitive linear indexing. This is more than 100 times faster than the first implementation.  
***Conv*:** The only difference between this and *Conv_3* is parallelization. This is faster than *Conv_3* only on multicore CPUs. It takes no more than 1-2 seconds to pass an image through all the layers using *Conv* or *Conv_3* on a single CPU core. A faster generalized matrix multiplication approach can make it even faster...

## Links
The paper that proposes a faster convolution approach (used in *Conv_2*, *Conv_3* and *Conv*):  
https://arxiv.org/abs/1410.0759

Pretrained models (for matconvnet):  
http://www.vlfeat.org/matconvnet/pretrained/

Visual Geometry Group (VGG)  
http://www.robots.ox.ac.uk/~vgg/

The method that I used in *InputLayer* to read a bitmap quickly [in Turkish]:  
http://huseyinatasoy.com/Bitmapleri-Net-Catisi-Altinda-Hizlica-Isleme

My blog post about this library [in Turkish]:  
http://huseyinatasoy.com/CeNiN-Konvolusyonel-Yapay-Sinir-Agi-Kutuphanesi
