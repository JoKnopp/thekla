#!/usr/bin/python
# -*- coding: UTF-8 -*-

#This file is part of Thekla.
#TODO licence


"""
Module comment

..  moduleauthor:: Johannes Knopp <johannes@informatik.uni-mannheim.de>
"""

from __future__ import absolute_import
from __future__ import unicode_literals

__version__ = '0.1'
__author__ = 'Johannes Knopp <johannes@informatik.uni-mannheim.de>'
__copyright__ = '© Copyright 2011 Johannes Knopp'

from TopicModel import TopicModel as TM
from Document import Documents as DOCS
import Visualize
import Clustering

import logging
import optparse
import ConfigParser
import sys
import json
import os
import codecs

#the global logger
LOG = logging.getLogger()
LOG.setLevel(logging.DEBUG)

EXAMPLECONF = '../example/example.conf'
DEFAULTS = {'centroid_computation':'avg', 'cluster_algorithm':'dbscan',
            'nwords':'5'}


def init_optionparser():
    """Initialise command line parser."""

    usage = 'Usage: %prog [options] config.txt'

    parser = optparse.OptionParser(usage)

    parser.add_option('-q', '--quiet',
        action='store_true',
        dest='quiet',
        default=False,
        help='Ignore informal output'
    )

    parser.add_option('-d', '--debug',
        default=False,
        action='store_true',
        dest='debug',
        help='Print debug output [default: %default]'
        )

    #Logging related options
    log_options = optparse.OptionGroup(parser,
            'Logging',
            'Specify log file handling.'
    )

    log_level = ['DEBUG', 'INFO', 'WARNING', 'ERROR']

    log_options.add_option('--log-file',
                            metavar='FILE',
                            type='string',
                            help='write logs to FILE'
                            )

    log_options.add_option('--log-file-level',
                            help='set log level (' +
                            ', '.join(log_level) +
                            ') [default: %default]',
                            action='store', default='INFO',
                            type='choice', choices=log_level
                            )

    parser.add_option_group(log_options)
    return parser

def init_logging(options):
    """Initialise logging framework

    :param options: Options obtained from optparse"""

    error = logging.StreamHandler(sys.stderr)
    error.setLevel(logging.ERROR)
    error.formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    LOG.addHandler(error)

    if not options.quiet:
        loglevel = logging.INFO
        console = logging.StreamHandler()
        if options.debug:
            loglevel = logging.DEBUG
        console.setLevel(loglevel)
        console.formatter = logging.Formatter('[%(levelname)s]: %(message)s')
        LOG.addHandler(console)

    if options.log_file:
        log_file_handler = logging.FileHandler(
            options.log_file)
        log_file_handler.setLevel(
            logging.getLevelName(options.log_level))
        log_file_handler.formatter = logging.Formatter(
            '[%(levelname)s]: %(message)s')
        LOG.addHandler(log_file_handler)

    LOG.debug('Logging initialised')


def load_topic_model(config):
    """Loads a topic model."""
    vfile = config.get('topicmodel', 'vocabfile')
    bfile = config.get('topicmodel', 'betafile')
    LOG.info('loading topic model from "{file}"..'.format(file=bfile))
    if not os.path.exists(bfile):
        LOG.error('file {file} does not exist'.format(file=bfile))
        return None
    tm = TM(vfile, bfile)
    LOG.info('done')
    return tm


def represent_documents(config, tm):
    """Loads the documents and represents them using a tm."""
    doc_dir = config.get('documents', 'docdir')
    centroid_computation = config.get('documents', 'centroid_computation')
    LOG.info('representing documents from "{doc}" with topics..'.format(doc=doc_dir))
    documents = DOCS(doc_dir, tm, centroid_computation)
    LOG.info('done')
    return documents


def represent_documents_from_files(config, tm, files):
    """Loads the documents and represents them using a tm."""
    centroid_computation = config.get('documents', 'centroid_computation')
    LOG.info('representing {doccount} documents with topics..'.format(doccount=len(files)))
    documents = DOCS(files, tm, centroid_computation)
    LOG.info('done')
    return documents


def load_clustering(config, clustering_title):
    """Loads a config file containing document clusters."""
    cluster_file = config.get('clustering', 'clusterfile')
    LOG.info('loading clusters from "{file}"..'.format(file=cluster_file))
    clustering = ConfigParser.ConfigParser()
    clustering.read(cluster_file)
    clusters = {}
    for sec in clustering.sections():
        #must be absolute filenames
        doc_filenames = json.loads(clustering.get(sec, 'docs'))
        clusters[sec] = doc_filenames
    LOG.info('done')
    return clusters


def create_cluster_centroids(clusters, documents):
    """Returns a dictionary {cluster_name : cluster_centroid}."""
    LOG.info('computing cluster centroids')
    cluster_centroids= {}
    for c_name, cluster in clusters.iteritems():
        cluster_centroids[c_name] = documents.compute_cluster_centroid(cluster)
    LOG.info('done')
    return cluster_centroids


def create_clustering(documents, config):
    """Returns a clustering of the documents."""
    cluster_alg = config.get('clustering', 'cluster_algorithm')

    #start dbscan and infer the options
    if cluster_alg == 'dbscan' and not config.has_option('clustering','cluster_options'):
        clusters = Clustering.create_clusters_dbscan_infer_options(documents)
    #use given options
    else:
        cluster_options = json.loads(config.get('clustering','cluster_options'))
        clusters = Clustering.create_clusters(documents, cluster_alg,
                cluster_options)
    return clusters


def get_axis_labels(tm, nwords=5):
    """Returns a list of strings each describing one topic with its top words."""
    topic_top_words = tm.top_topic_words(nwords=nwords) #needed as axis labels
    topic_top_words = ['\n'.join(top_words) for top_words in topic_top_words]
    return topic_top_words


def visualize_clusters(config, tm, clustering_title, cluster_centroids):
    """Create radar chart for the given clustering and save it to a file."""
    res_dir = config.get('visualization', 'res_dir')
    if not res_dir.endswith(os.sep):
        res_dir += os.sep

    nwords = config.getint('visualization', 'nwords')
    axis_labels = get_axis_labels(tm, nwords)
    #save the image
    res_file = res_dir + '{title}.png'.format(title=clustering_title)
    Visualize.save_radar_chart(clustering_title, cluster_centroids,
            axis_labels,
            fname=res_file)


def export_to_semeval_format(config, documents):
    """Exports the clustering information to a file in the format used by the
    semeval evaluation of task 14 in Semeval 2010."""
    if not config.has_section('semeval'):
        return
    else:
        semeval_res_file = config.get('semeval', 'resfile')
        logging.info('exporting cluster result to {resfile}'.format(
                    resfile=semeval_res_file))
        if config.has_option('semeval', 'pos'):
            pos = config.get('semeval', 'pos')
            #check for correct input
            try:
                assert pos in ['v','n']
            except AssertionError as ae:
                LOG.error('could not export to semeval format, given pos tag must be either "v" or "n"')
                return
        else:
            pos = 'v' #default

    with codecs.open(semeval_res_file, 'a', 'utf-8') as rfile:
        for doc in documents.docs.itervalues():
            assignment = doc.cluster_assignment
            if assignment == -1:
                continue
            #fname example: /home/jknopp/code/thekla/example/data/deploy/deploy_1.txt
            word = os.path.basename(doc.fname).split('_')[0]
            index = os.path.basename(doc.fname).split('_')[1].split('.')[0]
            res_line = '{w}.{pos} {w}.{pos}.{i} {w}.{pos}.{k}\n'.format(
                    w=word, i=index, k = assignment, pos=pos)
            rfile.write(res_line)
            doc.set_cluster_assignment(-1) #reset clustering


def run_config(config_file):
    """Load data and generate cluster visualization from /config_file/."""
    config = ConfigParser.ConfigParser(DEFAULTS)
    config.read(config_file)

    #TODO figure out what has (not) to be done from the config file

    #we just start with loading a topic model
    tm = load_topic_model(config)
    if not tm:
        LOG.info('aborting, error in {config}'.format(config=config_file))
        return

    #set title for the clustering
    if config.has_option('clustering', 'title'):
        clustering_title = config.get('clustering', 'title')
    else:
        clustering_title = os.path.basename(config_file)

    if config.has_option('clustering', 'clusterfile'):
        #load clustering
        clusters = load_clustering(config, clustering_title)
        docnames = []
        for cluster in clusters.itervalues():
            for doc in cluster:
                docnames.append(doc)

        #create document representation with the tm for files in the clustering
        documents = represent_documents_from_files(config, tm, docnames)

    else:
        #create document representation with the tm
        documents = represent_documents(config, tm)

        #cluster using custom algorithm
        LOG.info('no clustering given, will create clustering myself..')
        #clustering_title += '_' + config.get('clustering','cluster_algorithm')
        clusters = create_clustering(documents, config)

    if len(clusters) == 0:
        LOG.info('no clusters for "{title}" found; aborting..'.format(
            title=clustering_title))
        return

    #represent clusters with their centroids
    cluster_centroids = create_cluster_centroids(clusters, documents)

    #export to semeval format
    export_to_semeval_format(config, documents)

    #visualize clusters
    visualize_clusters(config, tm, clustering_title, cluster_centroids)


def run_configs(config_files):
    #TODO remove semeval file if it exists
    for config_file in config_files:
        run_config(config_file)


if __name__ == '__main__':
    parser = init_optionparser()
    (options, args) = parser.parse_args()
    init_logging(options)
    config_list = [] #list of config files
    if args:
        if len(args) > 1:
            LOG.error('Wrong number of arguments')
            LOG.error(parser.usage)
        elif os.path.isfile(args[0]):
            config_list.append(args[0])
        elif os.path.isdir(args[0]):
            cdir = os.path.abspath(args[0]) + os.sep
            config_list = [cdir + d
                    for d in os.listdir(cdir)
                    if not os.path.isdir(cdir+d)]
            #ignore .swp files
            config_list = [c for c in config_list if not c.endswith('swp')]
    else:
        config_list.append(EXAMPLECONF)

    logging.debug('Working on the following config files: {clist}'.format(
        clist=config_list))
    run_configs(config_list)
