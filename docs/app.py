# app.py

'''
image2caption(image)
image2story(image)
image2romance(image)
image2reg(image, and form: pos, neg word)
'''
import cPickle as pkl
import pickle
from flask import Flask
from flask_cors import CORS
#import request
from flask import request
from flask import send_file
from flask import Flask, session, redirect, url_for, escape, request, Response
import os.path
import time
import uuid
import json

app = Flask(__name__)
CORS(app)

import pickle
import os
try:
    all_hashes = pickle.load(open('all_hashes.pkl', "rb"))
except:
    all_hashes = {}

try:
    all_hashes_caption = pickle.load(open('all_hashes_caption.pkl', "rb"))
except:
    all_hashes_caption = {}

try:
    all_hashes_getcaptions = pickle.load(open('all_hashes_getcaptions.pkl', "rb"))
except:
    all_hashes_getcaptions = {}

try:
    all_hashes_reg = pickle.load(open('all_hashes_reg.pkl', "rb"))
except:
    all_hashes_reg = {}



import cv2
#pip install imutils

def dhash(image, hashSize=8):
    # resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(image, (hashSize + 1, hashSize))
 
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
 
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

@app.route('/api/v1/image2romance', methods=['POST'])
def img2story():
    hexx = uuid.uuid4().hex
    file = request.files['image']
    filename = 'saved/'+hexx+'_'+file.filename #file.filename
    #print(filename)
    file.save(filename)

    # HASH > ALREADY DONE?
    imagecv = cv2.imread(filename)
    # if the image is None then we could not load it from disk (so
    # skip it)
    if imagecv is None:
        return '{"status": "error"}'
        # break
    # convert the image to grayscale and compute the hash
    imagecv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(imagecv)

    if imageHash in all_hashes:
        val = all_hashes[imageHash]
    else:    
        #write to file so theano knows it needs to work
        if os.path.isfile('flask.busy'):
            #theano or flask already busy, which it shouldnt so error
            return '{"status": "error"}'
        else:
            try:
                os.remove("output.txt")
            except OSError:
                pass
                
            #theano not busy, lets work
            f = open("input.txt","w+")
            f.write(filename)
            f.close() 

            #create lock file for flask
            f = open("flask.busy","w+")
            f.write("1")
            f.close() 
            waiting = True

            while waiting:
                time.sleep(0.01)
                #check if flask shouldnt wait anymore = result written to theano>output
                if not os.path.isfile('input.txt'):
                    waiting = False
            
            #return stuff in output.txt
            file = open("output.txt", "r") 
            val = file.read() 
            #print val
            
            all_hashes[imageHash] = val
            pickle.dump(all_hashes, open("all_hashes.pkl", "wb"))

            try:
                os.remove("flask.busy")
                os.remove("output.txt")
            except OSError:
                pass

    return '{"status": "ok", "response": '+json.dumps(val)+'}'

@app.route('/api/v1/image2story', methods=['POST'])
def img2coco():
    hexx = uuid.uuid4().hex
    file = request.files['image']
    filename = 'saved/'+hexx+'_'+file.filename #file.filename
    print(filename)
    file.save(filename)

    # HASH > ALREADY DONE?
    imagecv = cv2.imread(filename)
    # if the image is None then we could not load it from disk (so
    # skip it)
    if imagecv is None:
        return '{"status": "error"}'
        # break
    # convert the image to grayscale and compute the hash
    imagecv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(imagecv)

    if imageHash in all_hashes_caption:
        val = all_hashes_caption[imageHash]
    else:    
        #write to file so theano knows it needs to work
        if os.path.isfile('flask.busy'):
            #theano or flask already busy, which it shouldnt so error
            return '{"status": "error"}'
        else:
            try:
                os.remove("output_caption.txt")
            except OSError:
                pass
                
            #theano not busy, lets work
            f = open("input_caption.txt","w+")
            f.write(filename)
            f.close() 

            #create lock file for flask
            f = open("flask.busy","w+")
            f.write("1")
            f.close() 
            waiting = True

            while waiting:
                time.sleep(0.01)
                #check if flask shouldnt wait anymore = result written to theano>output
                if not os.path.isfile('input_caption.txt'):
                    waiting = False
            
            #return stuff in output.txt
            file = open("output_caption.txt", "r") 
            val = file.read() 
            #print val
            
            all_hashes_caption[imageHash] = val
            pickle.dump(all_hashes_caption, open("all_hashes_caption.pkl", "wb"))

            try:
                os.remove("flask.busy")
                os.remove("output_caption.txt")
            except OSError:
                pass

    return '{"status": "ok", "response": '+json.dumps(val)+'}'


@app.route('/api/v1/image2caption', methods=['POST'])
def img2caption():
    hexx = uuid.uuid4().hex
    file = request.files['image']
    filename = 'saved/'+hexx+'_'+file.filename #file.filename
    print(filename)
    file.save(filename)

    # HASH > ALREADY DONE?
    imagecv = cv2.imread(filename)
    # if the image is None then we could not load it from disk (so
    # skip it)
    if imagecv is None:
        return '{"status": "error"}'
        # break
    # convert the image to grayscale and compute the hash
    imagecv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(imagecv)

    if imageHash in all_hashes_getcaptions:
        val = all_hashes_getcaptions[imageHash]
    else:    
        #write to file so theano knows it needs to work
        if os.path.isfile('flask.busy'):
            #theano or flask already busy, which it shouldnt so error
            return '{"status": "error"}'
        else:
            try:
                os.remove("output_getcaptions.txt")
            except OSError:
                pass
                
            #theano not busy, lets work
            f = open("input_getcaptions.txt","w+")
            f.write(filename)
            f.close() 

            #create lock file for flask
            f = open("flask.busy","w+")
            f.write("1")
            f.close() 
            waiting = True

            while waiting:
                time.sleep(0.01)
                #check if flask shouldnt wait anymore = result written to theano>output
                if not os.path.isfile('input_getcaptions.txt'):
                    waiting = False
            
            #return stuff in output.txt
            file = open("output_getcaptions.txt", "r") 
            val = file.read() 
            #print val
            
            all_hashes_getcaptions[imageHash] = val
            pickle.dump(all_hashes_getcaptions, open("all_hashes_getcaptions.pkl", "wb"))
            
            try:
                os.remove("flask.busy")
                os.remove("output_getcaptions.txt")
            except OSError:
                pass

    return '{"status": "ok", "response": '+json.dumps(val)+'}'


@app.route('/api/v1/image2reg', methods=['POST'])
def img2regularity():
    pos = request.form['pos']
    neg = request.form['neg']
    hexx = uuid.uuid4().hex
    file = request.files['image']
    filename = 'saved/'+hexx+'_'+file.filename #file.filename
    print(filename)
    file.save(filename)

    # HASH > ALREADY DONE?
    imagecv = cv2.imread(filename)
    # if the image is None then we could not load it from disk (so
    # skip it)
    if imagecv is None:
        return '{"status": "error"}'
        # break
    # convert the image to grayscale and compute the hash
    imagecv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(imagecv)

    if imageHash in all_hashes_reg:
        val = all_hashes_reg[imageHash]
    else:    
        #write to file so theano knows it needs to work
        if os.path.isfile('flask.busy'):
            #theano or flask already busy, which it shouldnt so error
            return '{"status": "error"}'
        else:
            try:
                os.remove("output_regularity_1.txt")
            except OSError:
                pass
                
            #theano not busy, lets work
            f = open("input_regularity.txt","w+")
            f.write(filename)
            f.close() 
            f = open("input_regularity_pos.txt","w+")
            f.write(pos)
            f.close() 
            f = open("input_regularity_neg.txt","w+")
            f.write(neg)
            f.close() 

            #create lock file for flask
            f = open("flask.busy","w+")
            f.write("1")
            f.close() 
            waiting = True

            while waiting:
                time.sleep(0.01)
                #check if flask shouldnt wait anymore = result written to theano>output
                if not os.path.isfile('input_regularity.txt'):
                    waiting = False
            
            #return stuff in output.txt
            file = open("output_regularity_1.txt", "r") 
            val = file.read() 
            #print val
            
            all_hashes_reg[imageHash] = val
            pickle.dump(all_hashes_reg, open("all_hashes_reg.pkl", "wb"))

            try:
                os.remove("flask.busy")
                os.remove("output_regularity_1.txt")
            except OSError:
                pass

    return '{"status": "ok", "response": '+json.dumps(val)+'}'



### NEW ROUTES ###
'''
img2vec(im) > {"vector":[...]}
text2vector(txt) > {"vector":[...]}
path(v1,v2,hops=3) > [
    0: {"image": "http://url/someimg.jpg", "distance": 0.2, "vector": [...] },
    1: {"image": "http://url/someimg.jpg", "distance": 0.5, "vector": [...] },
    2: {"image": "http://url/someimg.jpg", "distance": 0.8, "vector": [...] },
]
vec2caption(v,max_neghbours=10) > [
    0: {"caption": "two lions in a field", "distance": 0.1, "vector": [...] },
    1: {"caption": "a lion is running", "distance": 0.1, "vector": [...] },
    2: {"caption": "some wild animals running", "distance": 0.1, "vector": [...] },
]
vec2images(v,max_neghbours=10) > [
    0: {"image": "http://url/someimg.jpg", "distance": 0.1, "vector": [...] },
    1: {"image": "http://url/someimg.jpg", "distance": 0.1, "vector": [...] },
    2: {"image": "http://url/someimg.jpg", "distance": 0.1, "vector": [...] },
]
vec2story(v) > {"story":"james was standing in the snow..."}
vec2romance(v) > {"story":"he lovingly kissed her..."}
'''

#wrapper for communciatign withb magic.py
def getMagic(request,name,text=False):
    try:
        hsh = pickle.load(open('hash_'+name+'.pkl', "rb"))
    except:
        hsh = {}
    
    if text:
        text = request.form['text']
        imageHash = text
        filename = text
    else:
        hexx = uuid.uuid4().hex
        file = request.files['image']
        filename = 'saved/'+hexx+'_'+file.filename #file.filename
        #print(filename)
        file.save(filename)

        imagecv = cv2.imread(filename)
        if imagecv is None:
            return '{"status": "error"}'
        
        imagecv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2GRAY)
        imageHash = dhash(imagecv)
    
    if imageHash in hsh:
        val = hsh[imageHash]
    else:    
        #write to file so theano knows it needs to work
        if os.path.isfile('flask.busy'):
            #theano or flask already busy, which it shouldnt so error
            return 'error' #'{"status": "error"}'
        else:
            try:
                os.remove("output_"+name+".txt")
            except OSError:
                pass
                
            #theano not busy, lets work
            f = open("input_"+name+".txt","w+")
            f.write(filename)
            f.close() 

            #create lock file for flask
            f = open("flask.busy","w+")
            f.write("1")
            f.close() 
            waiting = True

            while waiting:
                time.sleep(0.01)
                #check if flask shouldnt wait anymore = result written to theano>output
                if not os.path.isfile('input_'+name+'.txt'):
                    waiting = False
            
            #return stuff in output.txt
            file = open("output_"+name+".txt", "r") 
            val = file.read() 
            #print val
            
            hsh[imageHash] = val
            pickle.dump(hsh, open("hash_"+name+".pkl", "wb"))
            
            try:
                os.remove("flask.busy")
                os.remove("output_"+name+".txt")
            except OSError:
                pass

    return val #'{"status": "ok", "response": '+json.dumps(val)+'}'


@app.route('/api/v1/image2vec', methods=['POST'])
def img2vec():
    val = getMagic(request,'image2vec')
    if val == 'error':
        return '{"status": "error"}'
    else:
        return '{"status": "ok", "response": '+json.dumps(val)+'}'

@app.route('/api/v1/text2vec', methods=['POST'])
def text2vec():
    val = getMagic(request,'text2vec',text=True)
    if val == 'error':
        return '{"status": "error"}'
    else:
        return '{"status": "ok", "response": '+json.dumps(val)+'}'


@app.route('/api/v1/path', methods=['POST'])
def path():
    content = request.get_json(silent=True)
    v1 = content['v1']
    v2 = content['v2']
    if 'num_hops' in content:
        num_hops = content['num_hops']
    else:
        num_hops = 10

    if v1 is None:
        return '{"status": "error"}'
    if v2 is None:
        return '{"status": "error"}'
    if num_hops == None:
        num_hops = 10

    #no caching for now
    if True:
        #write to file so theano knows it needs to work
        if os.path.isfile('flask.busy'):
            #theano or flask already busy, which it shouldnt so error
            return '{"status": "error"}'
        else:
            try:
                os.remove("output_path.pkl")
            except OSError:
                pass
                
            #theano not busy, lets work
            with open('input_path1.pkl', 'wb') as fp:
                pickle.dump(v1, fp)
            with open('input_path2.pkl', 'wb') as fp:
                pickle.dump(v2, fp)
            with open('input_path3.pkl', 'wb') as fp:
                pickle.dump(num_hops, fp)

            #create lock file for flask
            f = open("flask.busy","w+")
            f.write("1")
            f.close() 
            waiting = True

            while waiting:
                time.sleep(0.01)
                #check if flask shouldnt wait anymore = result written to theano>output
                if not os.path.isfile('input_path1.pkl'):
                    waiting = False
            
            #return stuff in output.txt
            with open ('output_path.pkl', 'rb') as fp:
                val = pickle.load(fp)
            #print val
            
            try:
                os.remove("flask.busy")
                os.remove("output_path.pkl")
            except OSError:
                pass

    if val=='error':
        return '{"status": "error"}'

    return '{"status": "ok", "response": '+json.dumps(val)+'}'

from flask import send_from_directory

#@app.route('/img/<filename>')
@app.route('/<path:filename>')  
def uploaded_file2(filename):
    return send_from_directory('/home/ubuntu/EBS5/BAM',
                               filename)

import os
try:
    os.mkdir('static')
except:
    pass
try:
    os.mkdir('static/upload')
except:
    pass
try:
    os.mkdir('static/serve')
except:
    pass

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='8787')
