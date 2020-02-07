from os.path import join
import os

import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import normalize

class ExtractEmbeddingSimilarities(BaseEstimator,TransformerMixin):
    def __init__(self,emb_type='word2vec',
                 emb_dir='/10TBdrive/minje/features/embeddings',
                 method='average'):
        self.ex = ExtractWordEmbeddings(emb_type=emb_type,emb_dir=emb_dir,method=method)

        self.dim_words = {}
        self.ground_embedding = {}
        with open('lexicons/10_dimensions_seed_words.txt') as f:
            for line in f:
                dim,words = line.strip().split(':')
                dim = dim.strip()
                words = words.strip().split(',')
                self.dim_words[dim] = [w.strip() for w in words]

        for dim,words in self.dim_words.items():
            # option 1 - just add them without normalizing
            # self.ground_embedding[dim] = (self.ex.obtain_vectors_from_sentence(words, include_unk=False)).mean(0)

            # # option 2 - normalize them first, then add
            self.ground_embedding[dim] = normalize(self.ex.obtain_vectors_from_sentence(words, include_unk=False),norm='l2',axis=1).mean(0).squeeze()
        return

    def fit(self):
        return


    def transform_single(self, X):
        # transforms a single sentence into an embedding output
        if type(X)=='str':
            X = X.split()
        arr = self.ex.obtain_vectors_from_sentence(X).mean(0)
        out = arr.tolist()+[np.dot(arr,self.ground_embedding)]
        return out

    def transform(self, X):
        out = np.array([self.transform_single(x) for x in X])
        return out

    def get_feature_names(self):
        return ['emb:dim-%d'%i for i in range(len(self.ground_embedding))]+['emb:similarity']


# loads all pretrained word embeddings under the wv format using Gensim
class ExtractWordEmbeddings():
    def __init__(self,emb_type='glove',
                 emb_dir='weights/embeddings',
                 method='average'):
        from gensim.models import KeyedVectors

        emb_type = emb_type.lower()
        # if emb_type=='word2vec':
        #     load_dir = join(emb_dir,'word2vec/GoogleNews-vectors-negative300.wv')
        # elif emb_type=='fasttext':
        #     load_dir = join(emb_dir,'fasttext/wiki-news-300d-1M-subword.wv')
        if emb_type=='glove':
            # load_dir = join(emb_dir,'glove/glove.twitter.27B.200d.wv')
            load_dir = join(emb_dir,'glove.840B.300d.wv')

        self.model = KeyedVectors.load(load_dir,mmap='r')
        self.emb_type = emb_type
        self.method = method

        self.UNK = self.model.vectors.mean(0) # UNK as just the average of all vectors
        if emb_type=='word2vec':
            self.UNK = self.model['UNK']
        print("Loaded word embeddings from %s!"%load_dir)
        print("Vocab size: %d" %len(self.model.vocab))
        return

    def fit(self,X):
        return

    # from any sentence, returns word vectors
    def obtain_vectors_from_sentence(self, words, include_unk=True):
        out = []
        for word in words:
            if word in self.model:
                vec = self.model[word]
            elif word.lower() in self.model:
                vec = self.model[word.lower()]
            else:
                if include_unk:
                    vec = self.UNK
                else:
                    continue
            out.append(vec.tolist())
        if len(out)==0:
            return np.zeros(len(self.UNK)).reshape(1,-1)
        else:
            return np.array(out)

    def transform(self, X):
        """
        :param X: list containing
        :return:
        """
        #
        assert type(X) == list, "Error in ExtractTags: input is not a list!"
        assert type(X[0]) == list, "Error in ExtractTags: Input is not tokenized!"
        out = []
        # case 1: only 1 sentence per sample
        if type(X[0][0]) == str:
            for sentence in X:
                arr = self.obtain_vectors_from_sentence(sentence)  # sentence = list of words
                if self.method=='average':
                    arr = np.mean(arr,axis=0)
                out.append(arr)
        # case 2: each sample has multiple sentences
        elif type(X[0][0]) == list:
            if type(X[0][0][0]) == str:
                for sentences in X:
                    all_words = []
                    for sent in sentences:
                        all_words.extend(sent)
                    arr = self.obtain_vectors_from_sentence(all_words)  # sentence = list of words
                    if self.method == 'average':
                        arr = np.mean(arr, axis=0)
                    out.append(arr)
        return out

    def get_feature_names(self):
        return ['wv-%s-%s:%d'%(self.emb_type,self.method,i) for i in range(len(self.UNK))]


if __name__=='__main__':
    WV = ExtractWordEmbeddings(emb_type='glove')
    words = 'this is the test sentence I will try out'.split()
    vectors = WV.obtain_vectors_from_sentence(words)
    print(np.array(vectors))
    print(len(vectors))
    print(len(vectors[0]))