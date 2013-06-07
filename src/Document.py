from __future__ import absolute_import
from __future__ import unicode_literals

import logging
import codecs
import os

import scipy

def repr_with_tm(fname, tm):
    """
    Read file /fname/ and map each word to the id of the corresponding topic
    vector in /tm/ and save the result in a list. With the list and the topic
    model you can construct a word_id_list of topic vectors representing the
    document.
    """
    #!Warning!: The resulting word_id_list is bound to the topic models vocabulary.
    #Work with the ids only if you are sure you are working with the same
    #tm.vocab
    with codecs.open(fname, 'r', 'utf-8') as doc_file:
        for line in doc_file:
            if line.isspace():
                continue
            word_id_list = [] #stores the tm ids of the words in the document
            for word in line.split():
                #word = word.lower()
                if tm.word_id_dict.has_key(word):
                    word_id_list.append(tm.word_id_dict[word])
        if not word_id_list:
            logging.debug('Found no known word in file {fname}'.format(fname=fname))
        return word_id_list


def _normalize_vector(vec):
    """
    Normalize /vec/ (vector or matrix) to a range between 0 and 1.
    """
    #We are dealing with log probabilities, therefore all values are negative
    v_min = scipy.amin(vec)
    v_max = scipy.amax(vec)
    f = lambda x: (x-v_min)/(v_max-v_min)
    return f(vec)


class Documents:
    """Represent Documents using topic models"""

    def __init__(self, doc_source, tm, centroid_computation):
        self.tm = tm
        #load absolute file names of text files
        fnames = []
        if isinstance(doc_source, str):
            doc_dir = os.path.abspath(doc_source) + os.sep
            fnames = [doc_dir + doc for doc in os.listdir(doc_dir) if
                    doc.endswith('.txt')]
        elif isinstance(doc_source, list):
            raw_fnames = []
            for doc in doc_source:
                base = os.path.basename(doc)
                if base.endswith('.txt'):
                    doc_dir = os.path.abspath(doc)
                    fnames.append(doc_dir)

        #represent documents with help of topics
        self.docs = {}
        for fname in fnames:
            doc = Document(fname, tm, centroid_computation)
            if doc.centroid is not None:
                self.docs[fname] = doc


    def as_centroid_matrix(self):
        """
        Returns a matrix consisting of the document centroids and a list
        holding the information which document's centroid is in which line of
        the matrix.
        """
        id_list = []

        assert len(self.docs) > 0
        doc_centroid_matrix = scipy.zeros((0,len(self.tm)))
        for key, doc in self.docs.iteritems():
            id_list.append(key)
            doc_centroid_matrix = scipy.vstack(
                    (doc_centroid_matrix, doc.centroid))
        return doc_centroid_matrix, id_list


    def compute_cluster_centroid(self, doc_list, flavor='avg'):
        """
        Iterates over the doc_list and returns the average of all document's
        centroids.
        """
        cluster_centroid = scipy.zeros((1,len(self.tm)))
        for doc in doc_list:
            #add document centroid to the matrix
            try:
                centroid = self.docs[doc].centroid
            except KeyError as ke:
                logging.error(ke)
                logging.warning('centroid missing for document "{doc}"'.format(doc=doc))
                continue
            if centroid != None:
                cluster_centroid += centroid
        cluster_centroid /= len(doc_list)
        return cluster_centroid


class Document:
    """Represent a Document"""
    def __init__(self, fname, tm, centroid_flavor='avg'):
        #define which way is used to compute the centroid of the document
        flavors = {'avg' : self.compute_centroid_avg,
                'exp' : self.compute_centroid_avg_experimental}

        self.fname = fname
        self.cluster_assignment = -1
        compute_centroid = flavors[centroid_flavor]
        word_id_list = repr_with_tm(fname, tm)
        self.centroid = compute_centroid(tm, word_id_list)
        del(word_id_list)


    def __str__(self):
        return 'Document "{name}"'.format(name=self.fname)


    def set_cluster_assignment(self, k):
        self.cluster_assignment = k


    def compute_centroid_avg(self, tm, word_id_list):
        """
        Computes the normalized average of the topic vectors of all words in
        this document
        """
        try:
            #no centroid without a vector
            assert len(word_id_list) > 0
        except AssertionError as ae:
            logging.error(ae)
            logging.error('document "{name}" seems to be empty!'.format(name=self.fname))
            return None

        #create vector to sum words' topic vectors
        centroid = scipy.zeros(len(tm.topics))
        for w_id in word_id_list:
            topic_vector = tm.get_topic_vector_for_id(w_id)
            centroid += topic_vector
        #compute the average of the topics
        centroid = centroid / float(len(word_id_list))
        #normalize the vector to a range between 0 and 1
        centroid = _normalize_vector(centroid)
        return centroid


    def compute_centroid_avg_experimental(self, tm, word_id_list):
        """
        Computes the average of the *normalized* topic vectors of all words in
        this document
        """
        try:
            #no centroid without a vector
            assert len(word_id_list) > 0
        except AssertionError as ae:
            logging.error(ae)
            logging.error('document "{name}" seems to be empty!'.format(name=self.fname))
            return None

        #create vector to sum words' topic vectors
        centroid = scipy.zeros(len(tm.topics))
        #TODO if this is used regularly, the normalized topic vector could
        #already be precomputed in tm
        for w_id in word_id_list:
            topic_vector = tm.get_topic_vector_for_id(w_id)
            #normalize topic vector
            topic_vector = _normalize_vector(topic_vector)
            centroid += topic_vector
        #compute the average of the topics
        centroid = centroid / float(len(word_id_list))
        return centroid
