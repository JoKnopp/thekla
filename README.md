Thekla
======

Thekla is a python tool to visualize clusters of documents with the help of
topic modeling. You are free to use the software in any way you want but if you
do please cite the following paper:
Johannes Knopp and Johanna Voelker. Thekla -- Visual Semantic Analysis of Document Clusters; (Demo paper) presented at 25th Conference of the German Society for Computational Linguistics (GSCL 2013) ([PDF](http://gscl2013.ukp.informatik.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/conferences/gscl2013/demo_thekla.pdf))

Requirements
------------
Thekla relies on Python 2.7 and makes use of the libraries
[matplotlib](http://matplotlib.org/index.html) and
[scikit-learn](http://scikit-learn.org/). Thekla does not take care of creating
the topic model that is needed for the visualization! You need to feed the
output of David Blei's tool [lda-c](http://www.cs.princeton.edu/~blei/lda-c/)
to the program.

Demo
----

After downloading the source code and installing the required libraries you can
just start the example:

```
~/thekla/src$ python thekla.py
[INFO]: loading topic model from "../example/topic_model/deploy/final.beta"..
[INFO]: done
[INFO]: loading clusters from "../example/clustering/deploy.cluster"..
[INFO]: done
[INFO]: representing 20 documents with topics..
[INFO]: done
[INFO]: computing cluster centroids
[INFO]: done
saving "../example/result/deploy.png"
```

Go to the result directory and have a look at the picture! To understand what
is shown read the paper cited at the beginning of this document.

Configuration files
-------------------
The main configuration file is seperated into four parts:
1. _topicmodel_: Information to load the topic model
2. _documents_ : Information on the corpus
3. _clustering_: Information on the clustering of the corpus
4. _visualization_: Output related options

This commented configuration file should be self explaining

```
#comments start with "#"
#configuration sections are marked by brackets
[topicmodel]
#the vocabulary of the topic model
vocabfile = path/to/vocab/file.vocab
#the output of ldac
betafile = path/to/ldac/result/final.beta

[documents]
#directory where the text documents lie. We assume one file per document.
docdir = path/to/corpus/

[clustering]
#title used in the visualization
title = cluster_title
#file with the clusters that will be visualized
clusterfile = path/to/result.cluster

[visualization]
res_dir = path/to/save/resulting/visualizations/
#number of best topic words to be displayed
nwords = 7
```

So what is the __clusterfile__? This is the file holding the clustering
information of the documents that are stored in the directory identified by
__docdir__. It has a section marked by [Name] for each cluster. Each section
has a variable where a list of filenames that belong to that cluster is given:

```
#one section per cluster
[cluster1]
#list of documents in cluster1
docs: ["path/to/corpus/text_1.txt",
  "path/to/corpus/text_3.txt",
  "path/to/corpus/text_7.txt"]

[cluster2]
#list of documents in cluster2
docs: ["path/to/corpus/text_2.txt",
  "path/to/corpus/text_4.txt",
  "path/to/corpus/text_8.txt"]

[..]
```


Clustering Demo
---------------
Despite stating that thekla takes a document clustering as the input and
visualizes it, there is also some functionality to cluster the documents if no
cluster information is given. Try out the second example:
    
```
~/thekla/src$ python thekla.py ../example/example_clustering.conf
[INFO]: loading topic model from "/home/jknopp/code/thekla/example/topic_model/deploy/final.beta"..
[INFO]: done
[INFO]: representing documents from "/home/jknopp/code/thekla/example/data/deploy/" with topics..
[INFO]: done
[INFO]: no clustering given, will create clustering myself..
[INFO]: Cluster algorithm is DBSCAN; options estimation is {u'min_samples': 21.88, u'eps': 4.67}
[INFO]: Could not cluster 491 documents
[INFO]: computing cluster centroids
[INFO]: done
[INFO]: exporting cluster result to /home/jknopp/code/thekla/example/result/clusters.key
saving "/home/jknopp/thekla/example/result/deploy_clustering.png"
```

In this case the clustering was not predefined but found by the DBSCAN
algorithm. The result in this example is not really meaningful but maybe it is
for your data. Please refer to [this paper](http://link.springer.com/chapter/10.1007/978-3-642-40722-2_10#) to learn more about how the clustering is done using topic modeling.
    
Besides DBSCAN you can use kmeans clustering. Adjust the __clustering__ section
of the config file to configure the algorithms:

```
[clustering]
#display title
title = deploy_clustering

#cluster_algorithm - the algorithm used for clustering if no clustering is
#   given. possible values are dbscan (default) and kmeans

#cluster_options - a dictionary holding the options for the mentioned
#   clustering. Available options depend on the chosen algorithm, learn more at
#   http://scikit-learn.org/dev/modules/clustering.html
#Example:
#cluster_algorithm = dbscan
#cluster_options = {"eps":0.8, "min_samples":2}
#cluster_algorithm = kmeans
#cluster_options = {"k":3}
```
