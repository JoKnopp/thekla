from __future__ import absolute_import
from __future__ import unicode_literals

import codecs
import scipy
import logging

class TopicModel:
    """Represent a topic model."""

    def __init__(self, vocab_filename, beta_file):
        #array of words where words' position in the array is also their global id
        self.vocab = self.load_vocab(vocab_filename)
        #a dictionary that maps words to their row index in the topics
        self.word_id_dict = self._create_word_id_dict()
        #matrix representation of topics. Each line represents one topic, each
        #row representes one word
        self.topics = self.create_topic_word_matrix_from_betafile(beta_file)


    def __str__(self):
        """Return the top words for each topic"""
        res_str = ''
        topic_repr = self.top_topic_words()
        for topic_no, topic_top_words in enumerate(topic_repr):
            res_str += 'topic {:>3}\n'.format(topic_no+1)
            for top_word in topic_top_words:
                res_str += '   {0}\n'.format(top_word)
            res_str += '\n'
        return res_str


    def __len__(self):
        return len(self.topics)


    def load_vocab(self, vocab_filename):
        """
        Loads the given file assuming there is one word per line and the line
        is the word's id. Returns the respective information as an array where
        the id of the word is it's index in the array.
        """
        try:
            with codecs.open(vocab_filename, 'r', 'utf-8') as vfile:
                vocab = [v.strip() for v in vfile]
        except IOError as e:
            logging.ERROR(e)
            raise
        return vocab


    def create_topic_word_matrix_from_betafile(self, beta_file):
        """
        Loads beta_file (stemming from an lda-c run) and represents the contained
        topics as a matrix in a scipy ndarray. Each line represents one topic, each
        row represents a word.
        """
        try:
            with open(beta_file, 'r') as bfile:
                #create first array entry
                topic_no = 0
                topic = bfile.readline()
                topic = map(float, topic.split())
                topics = scipy.array(topic)

                #add the rest of the topics to the array
                for topic in bfile:
                    topic = map(float, topic.split())
                    topics = scipy.vstack((topics,topic)) #append line
        except IOError as e:
            logging.ERROR(e)
            raise

        return topics


    def _create_word_id_dict(self):
        """Returns a dictionary mapping words to their IDs"""
        word_id_dict = {}
        for w_id, word in enumerate(self.vocab):
            word_id_dict[word] = w_id
        return word_id_dict


    def top_topic_words(self, nwords = 10):
        """Returns the /nwords/ highest ranked words in each topic."""
        topic_no = 0
        indices = range(len(self.vocab))
        res = []
        for topic in self.topics:
            #sort word indices by topic values
            indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
            topic_res = []
            for ii in range(nwords):
                topic_res.append(self.vocab[indices[ii]])
            res.append(topic_res)
            topic_no += 1
        return res


    def get_topic_vector_for_id(self, word_id):
        """
        Looks up the index of the given /word/ in the /vocab/ array in order to
        return the topic vector found in the /topic_word_matrix/
        """
        return self.topics[:,word_id]


    def get_topic_vector_for_word(self, word):
        """
        Looks up the index of the given /word/ in the /vocab/ array in order to
        return the topic vector found in the /topic_word_matrix/
        """
        try:
            ind = self.vocab.index(word) #get word's id
        except ValueError:
            return None
        return self.topics[:,ind]
