# KAGGLE_YELP

File run-inception-code.py uses <a href=https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md>Inception-V3 Network</a> to get TOP50 text features describing each photo.

To run it you need:

1) Copy all files in <b>model</b> directory from <a href=https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md>Inception repository</a>.

2) Put all JPG images for analysis in <b>../input/train_photos</b> folder

3) In case you have MXNet compiled with GPU, change ctx=mx.cpu() to ctx=mx.gpu() on line 12.
