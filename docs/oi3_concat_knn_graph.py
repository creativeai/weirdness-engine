import pickle
import glob

images = []
images2 = []
conv_feats = []
vse_feats = []

all_feats = {}

for loc in glob.glob('/home/ubuntu/EBS4/open-images-feats/*.p'):
    name = loc.split('/')[-1] #eg c7cadbcca13315e1.jpg.p
    name = '.'.join(name.split('.')[:-1])
    feat = pickle.load( open( loc, "rb" ) )
    conv_feats.append(feat)
    images.append(name)
    all_feats[name] = [feat,[0]]
print len(conv_feats),"conv feats loaded"

for loc in glob.glob('/home/ubuntu/EBS4/open-images-vse/*.p'):
    name = loc.split('/')[-1]
    name = name.split('.p.p')[0]
    feat = pickle.load( open( loc, "rb" ) )[0]
    vse_feats.append(feat)
    images2.append(name)
    conv_feat,_ = all_feats[name]
    all_feats[name] = [conv_feat, feat]
print len(vse_feats),"vse feats loaded"

conv_feats = []
vse_feats + []
for n in images2:
    conv_feats.append(all_feats[n][0])
    vse_feats.append(all_feats[n][1])
    
pickle.dump([images2, conv_feats], open('/home/ubuntu/EBS4/conv_feats_oid.p', 'wb'))
pickle.dump([images2, vse_feats], open('/home/ubuntu/EBS4/vse_feats_oid.p', 'wb'))
#same items in same order for vse and for conv

#and concat:
concat_feats = []
concat_names = []
for n in images2:
    concat_names.append(n)
    concat_feats.append(np.hstack((all_feats[n][0],all_feats[n][1])))

pickle.dump([concat_names, concat_feats], open('/home/ubuntu/EBS4/concat_feats_oid.p', 'wb'))

### ANNOY INDICES ###
# vse_feats_oid_annoy = for magic.py, text vector: find neigbour
# concat_feat_oid_annoy = for graph

# ANNOY INDEX
from annoy import AnnoyIndex

print("[!] Done extracting features, building search index")
ann_index = AnnoyIndex(len(vse_feats[0]))
for i in xrange(len(images)):
    ann_index.add_item(i, vse_feats[i])
    
print("[!] Constructing trees")
ann_index.build(100)
print("[!] Saving the index to '%s'" % 'vse_feats_oid_annoy')
ann_index.save('/home/ubuntu/EBS4/vse_feats_oid_annoy')
print("[!] Saving the filelist to '%s'" % ('vse_feats_oid_annoy' + ".filelist"))
filelist = file('/home/ubuntu/EBS4/vse_feats_oid_annoy' + ".filelist", "wt")
filelist.write("\n".join(images))
filelist.close()


# ANNOY INDEX
from annoy import AnnoyIndex

print("[!] Done extracting features, building search index")
ann_index = AnnoyIndex(len(concat_feats[0]))
for i in xrange(len(concat_names)):
    ann_index.add_item(i, concat_feats[i])
    
print("[!] Constructing trees")
ann_index.build(100)
print("[!] Saving the index to '%s'" % 'concat_feats_oid_annoy')
ann_index.save('/home/ubuntu/EBS4/concat_feats_oid_annoy')
print("[!] Saving the filelist to '%s'" % ('concat_feats_oid_annoy' + ".filelist"))
filelist = file('/home/ubuntu/EBS4/concat_feats_oid_annoy' + ".filelist", "wt")
filelist.write("\n".join(images))
filelist.close()

### GRAPH TIME ###

from scipy.spatial import distance
from igraph import *

#GRAPH USING ANN PART
kNN = 30
#graph = Graph(directed=False)
#graph.add_vertices(len(concat_names))
#graph.add_vertices(n=len(concat_names), directed=False, vertex_attrs={'name': concat_names})
graph = Graph(n=len(concat_names), directed=False, vertex_attrs={'name': concat_names})

for i in range(len(concat_names)):
    if i % 500 == 0:
        print("done %d / %d"%(i, len(concat_names)))
    
    #idx,dist = ann_index.get_nns_by_vector(concat_feats[0], 100,include_distances=True)
    idx = ann_index.get_nns_by_vector(concat_feats[i], 31,include_distances=False)
    #idx = idx[1:] #remove own
    #print idx
    #dist = dist[1:]
    feats_here = []
    idx_here = []
    for idd in idx:
        feats_here.append(concat_feats[idd])
        idx_here.append(idd)
        
    distances = [ distance.cosine(concat_feats[i], feat) for feat in feats_here ]
    #print distances
    idx_kNN = sorted(range(len(distances)), key=lambda k: distances[k])[1:kNN+1]
    #print idx_kNN #knn seems to work, no need for this, need distance thoygh
    
    for j in idx_kNN:
        dist = distances[j]
        j = idx[j]
        #print i,j,dist
        graph.add_edge(i, j, weight=dist)

pickle.dump([concat_names, graph], open('graph_oid.p', 'wb'))
