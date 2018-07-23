#THEANO_FLAGS='device=cuda0,floatX=float32' python magic.py

import pickle
import cPickle as pkl
import numpy
import copy
import sys
import skimage.transform

import skipthoughts
import decoder
import embedding

import config

import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
if not config.FLAG_CPU_MODE:
    from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer

from scipy import optimize, stats
from collections import OrderedDict, defaultdict, Counter
from numpy.random import RandomState
from scipy.linalg import norm

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time
import os
import json


def load_image(file_name):
    """
    Load and preprocess an image
    """
    MEAN_VALUE = numpy.array([103.939, 116.779, 123.68]).reshape((3,1,1))
    image = Image.open(file_name).convert('RGB')
    im = numpy.array(image)

    # Resize so smallest dim = 256, preserving aspect ratio
    if len(im.shape) == 2:
        im = im[:, :, numpy.newaxis]
        im = numpy.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = numpy.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = numpy.swapaxes(numpy.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUE
    return rawim, floatX(im[numpy.newaxis])

def compute_features(net, im):
    """
    Compute fc7 features for im
    """
    if config.FLAG_CPU_MODE:
        net.blobs['data'].reshape(* im.shape)
        net.blobs['data'].data[...] = im
        net.forward()
        fc7 = net.blobs['fc7'].data
    else:
        fc7 = numpy.array(lasagne.layers.get_output(net['fc7'], im,
                                                    deterministic=True).eval())
    return fc7

def build_convnet(path_to_vgg):
    """
    Construct VGG-19 convnet
    """
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    print 'Loading parameters...'
    output_layer = net['prob']
    model = pkl.load(open(path_to_vgg))
    lasagne.layers.set_all_param_values(output_layer, model['param values'])

    return net

import glob

print "loading net..."
net = build_convnet(config.paths['vgg'])

print "checking whats already done..."
import os.path
if os.path.isfile("/home/ubuntu/EBS4/oid1_done.p"):
    oid1_done = pickle.load( open( "/home/ubuntu/EBS4/oid1_done.p", "rb" ) )
else:
    oid1_done = []
    
if True:
    #scan to be sure
    for feat_loc in glob.glob('/home/ubuntu/EBS4/open-images-feats/*.p'):
        try:
            name = feat_loc.split('/')[-1]
            name = name.replace('.jpg.p','.jpg')
            vec = pickle.load( open( feat_loc, "rb" ) )
            oid1_done.append(name)
        except:
            pass #getting there in the loop next time
    pickle.dump( oid1_done, open( "/home/ubuntu/EBS4/oid1_done.p", "wb" ) )
print len(oid1_done)," images feats extracted already"

#images = []
print "extracting more conv features..."
for image_loc in glob.glob('/home/ubuntu/EBS5/OPENIMAGES/TRAIN/*.jpg'):
    name = image_loc.split('/')[-1]
    if not name in oid1_done:
        rawim, im = load_image(image_loc)
        # Run image through convnet
        feats = compute_features(net, im).flatten()
        feats /= norm(feats)
        pickle.dump( feats, open( "/home/ubuntu/EBS4/open-images-feats/"+name+".p", "wb" ) )




# Embed image into joint space
#feats = embedding.encode_images(z['vse'], feats[None,:])


