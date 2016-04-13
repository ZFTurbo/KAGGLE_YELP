# -*- coding: utf-8 -*-

import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
import glob
import json
import os

# Required: https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-bn.md

prefix = "Inception-BN/Inception_BN"
num_round = 39
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)
mean_img = mx.nd.load("Inception-BN/mean_224.nd")["mean_img"]

# load synset (text label)
synset = [l.strip() for l in open('Inception-BN/synset.txt').readlines()]


def PreprocessImage(path, show_img=False):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.multiply(resized_img, 256)
    # swap axes to make image from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - mean_img.asnumpy()
    # sample.resize(1, 3, 224, 224)
    return np.reshape(normed_img, (1, 3, 224, 224))

run_type = 2
if run_type == 1:
    train_index_json = './data/train_index_bn.json'
    path = os.path.join('..', 'initial_data', 'train_photos', '*.jpg')
else:
    train_index_json = './data/test_index_bn.json'
    path = os.path.join('..', 'initial_data', 'test_photos', '*.jpg')
files = glob.glob(path)

if (os.path.isfile(train_index_json)):
    f = open(train_index_json, 'r')
    out_index = json.load(f)
    f.close()
else:
    out_index = dict()

total = 0
for fl in files:
    flbase = os.path.basename(fl)
    if flbase in out_index.keys():
        print('File {} already processed: {}'.format(flbase, out_index[flbase]['0']))
        continue

    # Get preprocessed batch (single image batch)
    batch = PreprocessImage(fl)
    # Get prediction probability of 1000 classes from model
    prob = model.predict(batch)[0]
    # Argsort, get prediction index from largest prob to lowest
    pred = np.argsort(prob)[::-1]

    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print("File {} Top5: ".format(fl), top5)

    out_index[flbase] = dict()
    for i in range(50):
        out_index[flbase][i] = str(pred[i])
    # out_vals[flbase] = dict()
    # for i in range(50):
    #     out_vals[os.path.basename(fl)][i] = synset[pred[i]]
    total += 1

    # Save intermediate data
    if total % 5000 == 0:
        print('Saving now...')
        f = open(train_index_json, 'w')
        json.dump(out_index, f)
        f.close()
        # f = open(train_values_json, 'w')
        # json.dump(out_vals, f)
        # f.close()

print('Saving now...')
f = open(train_index_json, 'w')
json.dump(out_index, f)
f.close()
