# -*- coding: utf-8 -*-

import mxnet as mx
import numpy as np
from skimage import io, transform
import glob
import json
import os

prefix = "model/Inception-7"
num_round = 1
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)

# load synset (text label)
synset = [l.strip() for l in open('model/synset.txt').readlines()]

def PreprocessImage(path):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 299, 299
    resized_img = transform.resize(crop_img, (299, 299))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (299, 299, 3) to (3, 299, 299)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - 128.
    normed_img /= 128.

    return np.reshape(normed_img, (1, 3, 299, 299))

run_stage = 1
if not os.path.isdir(os.path.join('..', 'data')):
    os.mkdir(os.path.join('..', 'data'))
if run_stage == 1:
    train_index_json = os.path.join('..', 'data', 'train_index.json')
    path = os.path.join('..', 'input', 'train_photos', '*.jpg')
else:
    train_index_json = os.path.join('..', 'data', 'test_index.json')
    path = os.path.join('..', 'input', 'test_photos', '*.jpg')

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
    if total % 500 == 0:
        print('Saving now...')
        f = open(train_index_json, 'w')
        json.dump(out_index, f)
        f.close()

print('Saving now...')
f = open(train_index_json, 'w')
json.dump(out_index, f)
f.close()
