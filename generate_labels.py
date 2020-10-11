# -*- coding: utf-8 -*-
import os,sys
import csv
import nltk
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import random
import base64
from PIL import Image

def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

data_dir = '/raid_sdd/dataset/retrieval'

label_file_path = os.path.join(data_dir, 'training_public.csv')

images = []
sentences = []
with open(label_file_path)as f:
    f_csv = csv.reader(f)
    for i,row in enumerate(f_csv):
        if i == 0: continue
        sentences.append(row[1])
        images.append(row[0])

vectorizer = CountVectorizer(max_features=3000, ngram_range=(3,5), stop_words='english')
X = vectorizer.fit_transform(sentences).toarray()
label_file = 'label.txt'
feature_names = vectorizer.get_feature_names()
with open(label_file, 'w') as fw:
    for i, feat_name in enumerate(feature_names):
        print (i, feat_name, X[:,i].sum())
        fw.write(feat_name.encode('utf8')+'\t'+str(X[:,i].sum())+'\n')

fw = open('feature_3000_3_5.txt', 'w')
for i, img in enumerate(images):
    labels = [str(j) for j in np.where(X[i,:]>0)[0]]
    if len(labels) > 0:
        fw.write(img + '\t' + ','.join(labels)+'\n')
    
fw.close()
