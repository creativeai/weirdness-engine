#THEANO_FLAGS='device=cuda0,floatX=float32' python magic.py

"""
Story generation
"""
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
    
def text2vec(z,text):
    vec = embedding.encode_sentences(z['vse'], [text], verbose=False)
    # Compute skip-thought vectors for sentences
    #svecs = skipthoughts.encode(z['stv'], sentences, verbose=False)
    return vec

def image2vec(z,image_loc):
     # Load the image
    rawim, im = load_image(image_loc)

    # Run image through convnet
    feats = compute_features(z['net'], im).flatten()
    feats /= norm(feats)

    # Embed image into joint space
    feats = embedding.encode_images(z['vse'], feats[None,:])

    return feats

#images, vse_features = pickle.load( open( "vse_feats2.p", "rb" ) )
images, vse_features = pickle.load( open( "/home/ubuntu/EBS4/vse_feats_oid.p", "rb" ) )
#concat_pca_tsne_embedded2.p   #TSNE
#concat_pca_features2          #PCA
#concat_feats2.p               #concatted
#vse_feats2.p
#conv_feats2.p    
#images, conv_features = pickle.load( open( "concat_pca_tsne_embedded2.p", "rb" ) )
images, conv_features = pickle.load( open( "/home/ubuntu/EBS4/conv_feats_oid.p", "rb" ) )
#print('finished loading vse features for %d images' % len(images))

#images, graph = pickle.load( open( "graph_2.p", "rb" ) )
images, graph = pickle.load( open( "/home/ubuntu/EBS4/graph_oid.p", "rb" ) )


from scipy.spatial import distance
import igraph as igraph
#import uuid

# ANNOY INDEX
from annoy import AnnoyIndex
#ann_index = AnnoyIndex(1024)
#ann_index.load('concat_feats_annoy')
ann_index = AnnoyIndex(1024)
ann_index.load('/home/ubuntu/EBS4/vse_feats_oid_annoy')


def get_closest_images(query_image_feat, p_features, num_results=5):

    idx_closest,distances = ann_index.get_nns_by_vector(query_image_feat, num_results, include_distances=True)

    #distances = [ distance.euclidean(query_image_feat, feat) for feat in p_features ]
    #idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

import time

def get_image_path_between(query_image_idx_1, query_image_idx_2, feats_use, num_hops=100):
    #path = [query_image_idx_1, query_image_idx_2]

    #GRAPH PATH
    #TODO: hops now not used, as gets the shortest path
    #TODO: if more needed = add nearest neiugbours for each point
    print "get_image_path_between()",query_image_idx_1,query_image_idx_2
    start_millis = int(round(time.time() * 1000))
    if query_image_idx_1 == query_image_idx_2:
        print query_image_idx_1,"and",query_image_idx_2,"the same, taking random other index"
        #TODO: warning to user
        return "error","error"
    #mode 1 = OUT
    '''
    path = graph.get_shortest_paths(query_image_idx_1, to=query_image_idx_2, mode=igraph.OUT, output='vpath', weights='weight')[0]
    cur_weight = 0
    dist = []
    path_len = len(path)
    for i,e in enumerate(path):
        #print i
        if i < path_len-1:
            edgeid = graph.get_eid(path[i],path[i+1])
            edge = graph.es.select(edgeid)
            w = edge['weight'][0]
            #print w
            cur_weight += w
            dist.append(cur_weight)
    
    import numpy as np
    from sklearn.preprocessing import minmax_scale
    dist1 = minmax_scale(dist, feature_range=(0.1, 0.9))
    dist2 = np.array(dist1).tolist()
    dist2.insert(0,0)
    dist2.append(1)'''
    #new version adding neighbours
    #can run in loop until we get the nr we want with hops:
    #run path once, and then calculate how many neigbours wanted
    #extra_nn_idx=[]
    #extra_nn_dist=[]
    EXTRA = 0
    added = 0
    be_at = 0

    #REMAIN = 0 
    #rem=0 #if over 1 we add one
    path2 = []
    #test = 0
    #test2 = 0
    #testadded = 0
    if True:
        #path = graph.get_shortest_paths(query_image_idx_1, to=query_image_idx_2, mode=igraph.OUT, output='vpath', weights='weight')[0]
        path = graph.get_shortest_paths(query_image_idx_1, to=query_image_idx_2, 
            mode=igraph.ALL, output='vpath', weights='weight')[0]
        #path2 = path
        cur_weight = 0
        dist = []
        #dist.append(0)
        #path2.append(query_image_idx_1)
        path_len = len(path)
        if path_len < num_hops:
            EXTRA = float(num_hops)/float(path_len)
            #REMAIN = float(num_hops)/float(path_len)-EXTRA
            print "path nodes",path_len,"< requested hops of ",num_hops
            print "adding",EXTRA,"neighbours per path node"
        for i,e in enumerate(path):
            be_at += EXTRA
            #print i
            path2.append(e)
            added += 1
            #testadded += 1
            #print "testadded",testadded
            if i < path_len-1:
                edgeid = graph.get_eid(path[i],path[i+1])
                edge = graph.es.select(edgeid)
                w = edge['weight'][0]
                #print w
                cur_weight += w
                dist.append(cur_weight)
            else:
                dist.append(cur_weight)
                
            if EXTRA > 0:
                #to_get = rem
                #grab 5 in case one already exists
                #print "extra as neigbour of",i
                idx_closest,distances = ann_index.get_nns_by_vector(vse_features[path[i]], 50, include_distances=True)
                for i2,idx in enumerate(idx_closest): #add if it makes it where we should be, num_hops wise
                    if added+1 <= be_at:
                        if not idx in path:
                            added += 1
                            path2.append(idx)
                            dist.append(cur_weight)
    #end new method
    #print dist
    #print path2

    if len(dist)==0:
        #backup method
        print "no distance found backup slow method:..."
        #mode 1 = OUT
        
        path = graph.get_shortest_paths(query_image_idx_1, to=query_image_idx_2, mode=igraph.OUT, output='vpath', weights='weight')[0]
        cur_weight = 0
        dist = []
        path_len = len(path)
        for i,e in enumerate(path):
            #print i
            if i < path_len-1:
                edgeid = graph.get_eid(path[i],path[i+1])
                edge = graph.es.select(edgeid)
                w = edge['weight'][0]
                #print w
                cur_weight += w
                dist.append(cur_weight)
        
        import numpy as np
        from sklearn.preprocessing import minmax_scale
        dist1 = minmax_scale(dist, feature_range=(0.1, 0.9))
        dist2 = np.array(dist1).tolist()
        dist2.insert(0,0)
        dist2.append(1)
        
    else:
        from sklearn.preprocessing import minmax_scale
        dist2 = minmax_scale(dist, feature_range=(0, 1))
        path=path2

    '''
    #OTHER METHOD
    dist = []
    print "get_image_path_between()"
    start_millis = int(round(time.time() * 1000))
    #print(num_hops)
    for hop in range(num_hops-1):
        t = float(hop+1) / num_hops
        lerp_acts = t * feats_use[query_image_idx_1] + (1.0-t) * feats_use[query_image_idx_2]
        #print lerp_acts
        distances = [distance.euclidean(lerp_acts, feat) for feat in feats_use]
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])
        ix = [i for i in idx_closest if i not in path][0]
        dist.append(distances[ix])
        path.insert(1, [i for i in idx_closest if i not in path][0])

        end_millis = int(round(time.time() * 1000))
        print "hop",hop,": at ",(end_millis-start_millis),"ms"
    #dist.append(1)
    with open('dist0.pkl', 'wb') as fp:
        pickle.dump(dist, fp)
    
    from sklearn.preprocessing import minmax_scale
    dist1 = minmax_scale(dist, feature_range=(0.1, 0.9))
    with open('dist1.pkl', 'wb') as fp:
        pickle.dump(dist1, fp)
    import numpy as np
    dist2 = np.array(dist1).tolist()
    dist2.insert(0,0)
    dist2.append(1)
    #can be loaded for debugging
    with open('dist2.pkl', 'wb') as fp:
        pickle.dump(dist2, fp)

    '''
    end_millis = int(round(time.time() * 1000))
    print "returning total path length of",len(path)
    print "get_image_path_between() took",(end_millis-start_millis),"ms"
    
    return path,dist2

import uuid
from PIL import Image
def path(z,v1,v2,num_hops=10):
    print "path()"
    #v1 = np.asarray(v1)
    #v2 = np.asarray(v2)
    start_millis = int(round(time.time() * 1000))
    idx_closest1 = get_closest_images(v1,vse_features, num_results=1)
    idx_closest2 = get_closest_images(v2,vse_features, num_results=1)
    end_millis = int(round(time.time() * 1000))
    print "2x get_closest_images() took",(end_millis-start_millis),"ms"
    #try with conv_features here!
    path, dist = get_image_path_between(idx_closest1[0], idx_closest2[0], conv_features, num_hops)
    if path == "error":
        return "error","error"
    #moving files around
    start_millis2 = int(round(time.time() * 1000))
    #/home/ubuntu/EBS5/BAM/content_people/81db4417435165.562b9ee8bea7e.jpg
    #=
    #http://ec2-34-243-15-197.eu-west-1.compute.amazonaws.com:8787/content_people/81db4417435165.562b9ee8bea7e.jpg
    # sp
    filenames = []
    #filenames.append(filename1)
    for i,idx in enumerate(path):
        #img = Image.open(images[idx])
        #hexx = uuid.uuid4().hex
        #ext = images[idx].split('.')[-1]
        #filename = 'static/serve/'+hexx+'.'+ext
        #img.save(filename)
        filenames.append(images[idx].replace('/home/ubuntu/EBS5/BAM/',''))
    #filenames.append(filename2)
    end_millis2 = int(round(time.time() * 1000))
    print "movimg images around took",(end_millis2-start_millis2),"ms"

    end_millis = int(round(time.time() * 1000))
    print "total path() took",(end_millis-start_millis),"ms"
    return filenames,dist

def load_all(fast=False):
    """
    Load everything we need for generating
    """
    print config.paths['decmodel']

    if not fast:
        # Skip-thoughts
        print 'Loading skip-thoughts...'
        stv = skipthoughts.load_model(config.paths['skmodels'],
                                    config.paths['sktables'])

    if not fast:
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

    if not fast:
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
    if not fast:
        z['stv'] = stv
        z['dec'] = dec
    z['vse'] = vse
    z['net'] = net
    if not fast:
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

z = load_all(fast=True)

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
    
    elif os.path.isfile('input_text2vec.txt'): #got input to work with
        #generate_caption
        file = open("input_text2vec.txt", "r") 
        text = file.read() 
        out = text2vec(z, text)
        out = out.tolist()
        f = open("output_text2vec.txt","w+")
        f.write(json.dumps(out))
        f.close() 

        try:
            os.remove("input_text2vec.txt")
        except OSError:
            pass
    
    elif os.path.isfile('input_image2vec.txt'): #got input to work with
        #generate_caption
        file = open("input_image2vec.txt", "r") 
        imgname = file.read() 
        out = image2vec(z, imgname)
        out = out.tolist()
        f = open("output_image2vec.txt","w+")
        f.write(json.dumps(out))
        f.close() 

        try:
            os.remove("input_image2vec.txt")
        except OSError:
            pass

    elif os.path.isfile('input_path3.pkl'): #got input to work with
        #generate_caption
        #wait a bit so other files also written
        #import time
        #time.sleep(0.1)
        with open ('input_path1.pkl', 'rb') as fp:
            v1 = pickle.load(fp)
        with open ('input_path2.pkl', 'rb') as fp:
            v2 = pickle.load(fp)
        with open ('input_path3.pkl', 'rb') as fp:
            num_hops = pickle.load(fp)

        #out = path(z, v1, v2)
        ims, dist = path(z,v1,v2, num_hops=num_hops)
        if ims == "error":
            with open('output_path.pkl', 'wb') as fp:
                pickle.dump('error', fp)
        else:
            all_vals = []
            #{items:[{path:x,distance:2},{path:x,distance:2}], num_items:2}
            for i,p in enumerate(ims):
                vl = {}
                vl['path'] = str(p)
                vl['distance'] = dist[i]
                all_vals.append(vl)

            with open('output_path.pkl', 'wb') as fp:
                pickle.dump(all_vals, fp)

        try:
            os.remove("input_path1.pkl")
        except OSError:
            pass
        try:
            os.remove("input_path2.pkl")
        except OSError:
            pass
        try:
            os.remove("input_path3.pkl")
        except OSError:
            pass

    else:
        time.sleep(0.01)
        

