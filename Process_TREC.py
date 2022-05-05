import os
import os.path
import copy
import nltk
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import scipy.sparse as sp
from tokenizer import Tokenizer
import scipy.io as sio

class_num = 6

corpus = []
with open('./TREC/train_5500.label.txt', encoding="utf8", errors='ignore') as f:
    lines = f.readlines()
for line in lines:
    corpus.append(line.strip())
len_neg = len(corpus)
with open('./TREC/TREC_10.label.txt', encoding="utf8", errors='ignore') as f:
    lines = f.readlines()
for line in lines:
    corpus.append(line.strip())
print('neg_len: {}, pos_len :{}'.format(len_neg, len(corpus)-len_neg))

if class_num == 6:
    def clean_line(line):
        line = line.split(':')
        label = line[0]
        context = line[-1]
        return label, context

elif class_num == 50:
    def clean_line(line):
        line = line.split(' ')
        label = line[0]
        context = ''
        for word in line[1:]:
            context += word
            context += ' '
        return label, context


doc_label = []
doc_context = []
for doc in corpus:
    label, context = clean_line(doc)
    doc_label.append(label)
    doc_context.append(context)

label_set = set(doc_label)
label_name = list(label_set)
print(len(label_name), label_name)


vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=2000, tokenizer=Tokenizer.tokenize)
# vectorizer = CountVectorizer(lowercase=True, stop_words=stop_word, max_features=10000)
X = vectorizer.fit_transform(doc_context)
voc = vectorizer.vocabulary_
vectorizer = CountVectorizer(vocabulary=voc, tokenizer=Tokenizer.tokenize)
X = vectorizer.fit_transform(doc_context)
voc = vectorizer.get_feature_names()

data = X.toarray()
label = [label_name.index(name) for name in doc_label]
# np.random.seed(2020)
# np.random.shuffle(data)
# np.random.seed(2020)
# np.random.shuffle(label)

if class_num == 6:

    with open('TREC/trec_6.pkl', 'wb') as f:
        pickle.dump({'fea': data, 'gnd': label, 'voc': voc}, f)

    with open('TREC/trec_6.pkl', 'rb') as f:
        data = pickle.load(f)
    sio.savemat('TREC/trec_6.mat', data)

elif class_num == 50:

    with open('TREC/trec_50.pkl', 'wb') as f:
        pickle.dump({'fea': data, 'gnd': label, 'voc': voc}, f)

    with open('TREC/trec_50.pkl', 'rb') as f:
        data = pickle.load(f)
    sio.savemat('TREC/trec_50.mat', data)