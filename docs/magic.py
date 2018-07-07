#THEANO_FLAGS='device=cuda0,floatX=float32' python magic.py

"""
Story generation
"""
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

def regularities(z, image_loc, negword, posword, k=5, rerank=True):
    """
    This is an example of how the 'Multimodal Lingustic Regularities' was done.
    Returns nearest neighbours to 'image - negword + posword'
    z: the embedding model, with encoder='bow'
    z['net']: VGG ConvNet
    captions: a list of sentences
    imvecs: the corresponding image embeddings to each sentence in 'captions'
    image_loc: location of the query image
    negword: the word to subtract
    posword: the word to add
    k: number of results to return
    rerank: whether to rerank results based on their mean (to push down outliers)
    'captions' is used only as a reference, to avoid loading/displaying images.
    Returns:
    The top k closest sentences in captions
    The indices of the top k captions
    Note that in our paper we used the SBU dataset (not COCO)
    """
    
    # Load the image
    rawim, im = load_image(image_loc)

    # Run image through convnet
    query = compute_features(z['net'], im).flatten()
    query /= norm(query)

    # Embed words
    pos = embedding.encode_sentences(z['vse'], [posword], verbose=False)
    neg = embedding.encode_sentences(z['vse'], [negword], verbose=False)

    # Embed image
    query = embedding.encode_images(z['vse'], query[None,:])

    # Transform
    feats = query - neg + pos
    feats /= norm(feats)

    # Compute nearest neighbours
    scores = numpy.dot(feats, z['cvec'].T).flatten() #=imvecs.T which we dont got so got for cvecs for now
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [z['cap'][a] for a in sorted_args[:k]]

    # Re-rank based on the mean of the returned results
    if rerank:
        nearest = z['cvec'][sorted_args[:k]]
        meanvec = numpy.mean(nearest, 0)[None,:]
        scores = numpy.dot(nearest, meanvec.T).flatten()
        sargs = numpy.argsort(scores)[::-1]
        sentences = [sentences[a] for a in sargs[:k]]
        sorted_args = [sorted_args[a] for a in sargs[:k]]

    print sentences, sorted_args[:k] 
    return sentences, sorted_args[:k] 

def story(z, image_loc, k=100, bw=50, lyric=False, stochastic=False):
    """
    Generate a story for an image at location image_loc
    """
    # Load the image
    rawim, im = load_image(image_loc)

    # Run image through convnet
    feats = compute_features(z['net'], im).flatten()
    feats /= norm(feats)

    # Embed image into joint space
    feats = embedding.encode_images(z['vse'], feats[None,:])

    # Compute the nearest neighbours
    scores = numpy.dot(feats, z['cvec'].T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [z['cap'][a] for a in sorted_args[:k]]

    print 'NEAREST-CAPTIONS: '
    for s in sentences[:5]:
        print s
    print ''

    # Compute skip-thought vectors for sentences
    svecs = skipthoughts.encode(z['stv'], sentences, verbose=False)

    # Style shifting
    shift = svecs.mean(0) - z['bneg'] + z['bpos']

    # Generate story conditioned on shift
    passage = decoder.run_sampler(z['dec'], shift, beam_width=bw, stochastic=stochastic)
    return passage


def generate_caption(z, image_loc, k=100, bw=50, lyric=False, stochastic=False):
    """
    Generate a normal caption for an image at location image_loc
    """
    # Load the image
    rawim, im = load_image(image_loc)

    # Run image through convnet
    feats = compute_features(z['net'], im).flatten()
    feats /= norm(feats)

    # Embed image into joint space
    feats = embedding.encode_images(z['vse'], feats[None,:])

    # Compute the nearest neighbours
    scores = numpy.dot(feats, z['cvec'].T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [z['cap'][a] for a in sorted_args[:k]]

    print 'NEAREST-CAPTIONS: '
    for s in sentences[:5]:
        print s
    print ''

    # Compute skip-thought vectors for sentences
    svecs = skipthoughts.encode(z['stv'], sentences, verbose=False)

    # NO Style shifting
    shift = svecs.mean(0) #- z['bneg'] + z['bpos']

    # Generate story conditioned on shift
    passage = decoder.run_sampler(z['dec'], shift, beam_width=bw, stochastic=stochastic)
    return passage

def get_captions(z, image_loc, k=100, bw=50, lyric=False, stochastic=False):
    """
    Generate a normal caption for an image at location image_loc
    """
    # Load the image
    rawim, im = load_image(image_loc)

    # Run image through convnet
    feats = compute_features(z['net'], im).flatten()
    feats /= norm(feats)

    # Embed image into joint space
    feats = embedding.encode_images(z['vse'], feats[None,:])

    # Compute the nearest neighbours
    scores = numpy.dot(feats, z['cvec'].T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [z['cap'][a] for a in sorted_args[:k]]

    #print 'NEAREST-CAPTIONS: '
    #return '|'.join(sentences[:5])
    return sentences[:5]
    

def load_all():
    """
    Load everything we need for generating
    """
    print config.paths['decmodel']

    # Skip-thoughts
    print 'Loading skip-thoughts...'
    stv = skipthoughts.load_model(config.paths['skmodels'],
                                  config.paths['sktables'])

    # Decoder
    print 'Loading decoder...'
    dec = decoder.load_model(config.paths['decmodel'],
                             config.paths['dictionary'])

    # Image-sentence embedding
    print 'Loading image-sentence embedding...'
    vse = embedding.load_model(config.paths['vsemodel'])

    # VGG-19
    print 'Loading and initializing ConvNet...'

    if config.FLAG_CPU_MODE:
        sys.path.insert(0, config.paths['pycaffe'])
        import caffe
        caffe.set_mode_cpu()
        net = caffe.Net(config.paths['vgg_proto_caffe'],
                        config.paths['vgg_model_caffe'],
                        caffe.TEST)
    else:
        net = build_convnet(config.paths['vgg'])

    # Captions
    print 'Loading captions...'
    cap = []
    with open(config.paths['captions'], 'rb') as f:
        for line in f:
            cap.append(line.strip())

    # Caption embeddings
    print 'Embedding captions...'
    cvec = embedding.encode_sentences(vse, cap, verbose=False)

    # Biases
    print 'Loading biases...'
    bneg = numpy.load(config.paths['negbias'])
    bpos = numpy.load(config.paths['posbias'])

    # Pack up
    z = {}
    z['stv'] = stv
    z['dec'] = dec
    z['vse'] = vse
    z['net'] = net
    z['cap'] = cap
    z['cvec'] = cvec
    z['bneg'] = bneg
    z['bpos'] = bpos

    return z

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


z = load_all()
while True:

    #STORY
    if os.path.isfile('input.txt'): #got input to work with
        file = open("input.txt", "r") 
        imgname = file.read() 
        out = story(z, imgname)

        f = open("output.txt","w+")
        f.write(out)
        f.close() 

        try:
            os.remove("input.txt")
        except OSError:
            pass
    elif os.path.isfile('input_caption.txt'): #got input to work with
        #generate_caption
        file = open("input_caption.txt", "r") 
        imgname = file.read() 
        out = generate_caption(z, imgname)

        f = open("output_caption.txt","w+")
        f.write(out)
        f.close() 

        try:
            os.remove("input_caption.txt")
        except OSError:
            pass
    elif os.path.isfile('input_getcaptions.txt'): #got input to work with
        #generate_caption
        file = open("input_getcaptions.txt", "r") 
        imgname = file.read() 
        out = get_captions(z, imgname)

        f = open("output_getcaptions.txt","w+")
        f.write(json.dumps(out))
        f.close() 

        try:
            os.remove("input_getcaptions.txt")
        except OSError:
            pass
    elif os.path.isfile('input_regularity.txt'): #got input to work with
        #generate_caption
        file = open("input_regularity.txt", "r") 
        imgname = file.read() 
        file = open("input_regularity_pos.txt", "r") 
        pos = file.read() 
        file = open("input_regularity_neg.txt", "r") 
        neg = file.read() 
        sentences,args = regularities(z, imgname, neg, pos)
        print sentences, args

        f = open("output_regularity_1.txt","w+")
        f.write(json.dumps(sentences))
        f.close() 

        try:
            os.remove("input_regularity.txt")
            os.remove("input_regularity_pos.txt")
            os.remove("input_regularity_neg.txt")
        except OSError:
            pass
    else:
        time.sleep(5)

