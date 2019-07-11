#!/usr/bin/env python

"""The simplest TF-IDF library imaginable.

Add your documents as two-element lists `[docname,
[list_of_words_in_the_document]]` with `addDocument(docname, list_of_words)`.
Get a list of all the `[docname, similarity_score]` pairs relative to a
document by calling `similarities([list_of_words])`.

See the README for a usage example.

"""

import math

class TfIdf:
    def __init__(self):
        self.weighted = False
        self.documents = []
        self.tf = []
        self.idf= {}
        self.corpus_dict = {}
    
    def add_document(self, doc_name, list_of_words):
        """building a dictionary"""
        self.documents.append([doc_name, list_of_words])
        
    def calculate_tf(self):
        """calculate tf vectors for each doc"""
        for doc in self.documents:
            doc_dict = {}
            doc_name = doc[0]
            list_of_words = doc[1]
            for w in list_of_words:
                #updating docs having the word w; later to be used to calculate idf of word w
                if(doc_dict.get(w, 0.) == 0):
                    self.idf[w] = self.idf.get(w, 0.0) + 1.0
                else:
                    pass
                doc_dict[w] = doc_dict.get(w, 0.) + 1.0
                self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0 

            # normalizing the dictionary
            length = float(len(list_of_words))
            for k in doc_dict:
                doc_dict[k] = doc_dict[k] / length

            # add the normalized tf vector
            self.tf.append([doc_name, doc_dict])        
        
    def calculate_idf(self):
        """calculate idf vectors for each word"""
        length = float(len(self.documents))
        for k in self.idf:
            self.idf[k] = (1 + math.log(length/self.idf[k]) if self.idf[k] != 0 else 1)
        
    def calculate_tf_idf(self):
        """calculate tf_idf vectors for each doc"""
        for doc in self.tf:
            for k in doc[1]:
                doc[1][k] = doc[1][k]*self.idf[k]
    
    def cosine_similarity(self,v1,v2):
        """compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"""
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)


    def similarities(self, list_of_words):
        """Returns a list of all the [docname, similarity_score] pairs relative to a list of words."""

        # building the query dictionary
        query_dict = {}
        for w in list_of_words:
            query_dict[w] = query_dict.get(w, 0.0) + 1.0
 

        # normalizing the query for tf vector
        length = float(len(list_of_words))
        for k in query_dict:
            query_dict[k] = query_dict[k] / length
            
        # calculate tf_idf vector of query
        for k in query_dict:
            query_dict[k] = query_dict[k]*self.idf[k]

        # computing the list of similarities
        sims = []
        for doc in self.tf:
            doc_dict = doc[1]
            query = []
            doc_sc = []
            for k in query_dict:
                query.append(query_dict[k])
                if k in doc_dict:
                    doc_sc.append(doc_dict[k])
                else:
                    doc_sc.append(0)
            
            score = self.cosine_similarity(query,doc_sc)            
            sims.append([doc[0], score])
        return sims