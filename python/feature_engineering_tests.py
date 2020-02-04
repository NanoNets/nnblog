# unit tests

import os
from pathlib import Path
import json
from collections import namedtuple
import csv
import cv2

import re
from dateutil.parser import parse
# pip install geotext
from geotext import GeoText


# pip install bpemb
from bpemb import BPEmb
import numpy as np

# globals
# ---------------------------------------------------------------------------------------------------
from global_parameters import UNDEFINED, WordArea

# load English BPEmb model with default vocabulary size (10k) and 50-dimensional embeddings
bpemb_en = BPEmb(lang="en", dim=100)
# ---------------------------------------------------------------------------------------------------

from data_access import *
from graph_modeler import *
from feature_calculator import *


# -------------------------------------------------------------------------
# unit tests
# -------------------------------------------------------------------------
def graph_modeler_tests(normalized_dir, features_dir):
    print("\n\nRunning graph modeler tests")
    print('-'*100)
    
    try:
         # find files
        image_files, word_files, entity_files = get_normalized_filepaths(normalized_dir)
        # load one raw example
        image, word_areas, entities = load_normalized_example(image_files[0], word_files[0], entity_files[0])

        #test algo 1
        lines = line_formation(word_areas)

        # show lines
        for line in lines:
            print('\n')
            for tup in line:
                print(tup)

        # test algo 2
        image_width, image_height = cv_size(image)
        graph = graph_modeling(lines, word_areas, image_width, image_height)

        # plot image with annotations
        plot_image(image, word_areas, entities, graph, False)    
        print("Visualize Graph Data")
        print("writing {}".format("graph_example.png"))
        plt.savefig('graph_example.png')

        # adj matrix
        adjacency_matrix = form_adjacency_matrix(graph)
        print(adjacency_matrix.shape)
        print(np.sum(adjacency_matrix, axis=0))
        print(np.sum(adjacency_matrix, axis=1))
        print(np.sum(adjacency_matrix, axis=0) == np.sum(adjacency_matrix, axis=1))

        print(adjacency_matrix)
    except:
        return False
    
    return True


def feature_calculator_tests(normalized_dir, features_dir):
    print("\n\nRunning feature calculator tests")
    print('-'*100)
    
    try:
         # find files
        image_files, word_files, entity_files = get_normalized_filepaths(normalized_dir)
        # load one raw example
        image, word_areas, entities = load_normalized_example(image_files[0], word_files[0], entity_files[0])



        #test  1
        word_embeddings = [get_word_bpe(wa.content, bpemb_en) for wa in word_areas]
        print(len(word_embeddings[0]))

        #test  2
        binary_features = [get_word_binary_features(wa.content) for wa in word_areas]
        print(binary_features)

        #test 3
        # greate graph for numeric features
        lines = line_formation(word_areas)
        image_width, image_height = cv_size(image)
        graph = graph_modeling(lines, word_areas, image_width, image_height)    

        numeric_features = [get_word_area_numeric_features(wa, graph) for wa in word_areas]
        print(numeric_features)
    
    except:
        return False
    
    return True


# -------------------------------------------------------------------------
# Run the process in batch mode
# -------------------------------------------------------------------------
def main():

    # configs for data storage
    PROJECT_ROOT_PREFIX = os.path.abspath('../')
    NORMALIZED_PREFIX = "data/sroie2019/normalized/"
    normalized_dir = os.path.join(PROJECT_ROOT_PREFIX, NORMALIZED_PREFIX)
    FEATURES_PREFIX = "data/sroie2019/"
    features_dir = os.path.join(PROJECT_ROOT_PREFIX, FEATURES_PREFIX)


    results = []
    results.append(graph_modeler_tests(normalized_dir, features_dir))
    results.append(feature_calculator_tests(normalized_dir, features_dir))

    print("\n\nResults")
    print('-'*100)
    print("Total: {}    -    Pass: {}    -    Fail: {}".format(len(results), sum(results), len(results) - sum(results)))

if __name__ == "__main__":
    main()