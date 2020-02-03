# data_access.py

import os
from pathlib import Path
import json
from collections import namedtuple
import csv
import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------------
from global_parameters import UNDEFINED, WordArea

### Normalized Data Access
# ---------------------------------------------------------------------------------------------------
def get_filepaths(dataset_dir, file_type, extension):
    # return full file paths
    files = []
    subdir = os.path.join(dataset_dir, file_type)
    for r, d, f in os.walk(subdir):
        for filename in f:
            if filename.endswith(extension):
                files.append(os.path.join(subdir, filename))
    return sorted(files)
    
def get_normalized_filepaths(dataset_dir):
    # return full file paths
    image_files = get_filepaths(dataset_dir, "images", ".jpg")    
    word_files = get_filepaths(dataset_dir, "words", ".csv")
    entity_files = get_filepaths(dataset_dir, "entities", ".csv")
    
    return image_files, word_files, entity_files

def read_normalized_word_areas(words_filepath):
    """
    """                          
    word_areas = []
    with open(words_filepath, 'r', newline='') as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            word_area = WordArea(int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4], len(word_areas))
            word_areas.append(word_area)
    return word_areas

def read_normalized_entities(entities_filepath):
    entities = []
    with open(entities_filepath, 'r') as f:
        for line in f:
            entities.append(line.strip())
    return entities

def load_normalized_example(image_filepath, words_filepath, entities_filepath=''):
    """
    Reads one image and related data files from the normalized dataset.
    Reads the entities from text file in tabular format
    If the example is unlabeled, then entities_filepath can be omitted
    """
    word_areas, entities = [], []
    
    # load color (BGR) image
    # BGR is the same as RGB (just inverse byte order), OpenCV adopted it for historical reasons.
    image = cv2.imread(image_filepath)
    
    word_areas = read_normalized_word_areas(words_filepath)
    
    if len(entities_filepath) > 0:
        entities = read_normalized_entities(entities_filepath)
    
    return image, word_areas, entities

# ---------------------------------------------------------------------------------------------------
def create_filepath(parent_dir, dirname, filename):
    sub_dir = os.path.join(parent_dir,dirname)
    Path(sub_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(sub_dir, filename)


### Feature Access
# ---------------------------------------------------------------------------------------------------
def save_features(target_dir, image_filepath, 
                   word_embeddings=[], binary_features=[], numeric_features=[], debug=False):
    """
    save all the features computed for an image.
    all features and concatenated into a single feature vector and stored
    in a csv file with the name the image under the "features" directory.
    
    The graph is stored as json file, separately under a 'graph' directory.
    """
    image_filename = image_filepath.split('/')[-1].split(".")[0]
    
    # concatenate features
    features = [b + n + w for b, n, w in zip(binary_features, numeric_features, word_embeddings)]
    if debug: print(len(features), len(features[0]))
        
    # save features
    features_filename = "{}.csv".format(image_filename)
    features_filepath = create_filepath(target_dir, "features", features_filename)
    if debug: print(features_filepath)
    with open(features_filepath, 'w') as f:
        for feature_vector in features:
            f.write(",".join(["{}".format(x) for x in feature_vector]) + "\n")
        
def save_graph(target_dir, image_filepath, graph=[], adj_matrix=None):
    """
    save the graph computed for an image.
    The graph is stored as json file, separately under a 'graph' directory.
    """
    image_filename = image_filepath.split('/')[-1].split(".")[0]
        
    # save graph
    graph_filename = "{}.json".format(image_filename)
    graph_filepath = create_filepath(target_dir, "graph", graph_filename)
    with open(graph_filepath, 'w') as f:
        json.dump(graph, f)

    # save adj matrix
    adj_filename = "{}.csv".format(image_filename)
    adj_filepath = create_filepath(target_dir, "graph_adj_matrix", adj_filename)
    with open(adj_filepath, 'w') as f:
        for i in range(len(adj_matrix)):
            f.write(",".join(["{}".format(x) for x in adj_matrix[i,:]]) + "\n")
   
# ------------------------------------------------------------------------------------------------
def get_feature_filepaths(directory):
    # return full file paths
    
    feature_files = get_filepaths(directory, "features", ".csv")    
    graph_files = get_filepaths(directory, "graph", ".csv")
    adj_matrix_files = get_filepaths(directory, "graph_adj_matrix", ".csv")
    
    normalized_dir = os.path.join(directory, "normalized")
    entity_files = get_filepaths(normalized_dir, "entities", ".csv")
    
    return feature_files, graph_files, adj_matrix_files, entity_files


def read_features(features_filepath, adj_matrix_filepath):
    """
    Reads the features file and the graph_adj_matrix file for one image.
    Returns numpy arrays.
    
    Contents:
    There is a feature vector for each word_area of the image, this vector is
    a concatenation of binary, numerical and text features.
    The binary features contain information aabout the relative distance 
    to neighbour nodes.
    The graph adjacency matrix contains the connectivity structure of the
    graph.
    All the graph information is already available between the numeric 
    features and the adjacency matrix, so we don't need to read the graph file
    (the graph is stored as json file, separately under a 'graph' directory.)
    
    File format:
    both features and adj. matrix are stored in csv files.
    """
    # read features
    features = np.genfromtxt(features_filepath, delimiter=',', dtype=float) 
    
    # read adj matrix
    adj_matrix = np.genfromtxt(adj_matrix_filepath, delimiter=',', dtype=float) 
        
    return features, adj_matrix

   