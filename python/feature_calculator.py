# feature calculator

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

#### Text Features

def get_word_bpe(word, bpemb_en):
    """
    Get pretrained subword embeddings created with 
    the Byte-Pair Encoding (BPE) algorithm.
    If the word is segmented in more than 3 subwords we truncated at 3.
    The length of each subword embedding is 100.
    """
    
    # get subword embeddings
    embeddings = list(bpemb_en.embed(word))
    
    # trim size
    if len(embeddings) > 3:
        embeddings = embeddings[:3]
    elif len(embeddings) < 3:
        for _ in range(3 - len(embeddings)):
            embeddings.append(list(np.zeros(100)))

    # concatenate
    emb = list(embeddings[0])
    for e in embeddings[1:]:
        emb.extend(e)

    return emb

#### Binary Features
# -------------------------------------------------------------------------

def isDate(word, fuzzy=False):
    """
    Return whether the word can be interpreted as a date.

    :param word: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(word, fuzzy=fuzzy)
        return True
    except:
        return False

def isZipCode(word):
    return  re.match(r'(\b\d{5}-\d{4}\b|\b\d{5}\b\s)', word) is not None

def isKnownCity(word):
    return len(GeoText(word).cities) > 0

def isKnownState(word):
    return len(GeoText(word).country_mentions) > 0

def isKnownCountry(word):
    return len(GeoText(word).countries) > 0


def isAlphabethic(word):
    return word.isalpha()

def isNumeric(word):
    return word.isnumeric()

def isAlphaNumeric(word):
    return word.isalnum()

def isNumberWithDecimal(word):
    return  re.match(r'^[1-9]\d*\.\d*$', word)  is not None

def isRealNumber(word):
    try: 
        float(word)
    except ValueError: 
        return False
    return True

def isCurrency(word):
    # exactly 2 decimals
    return re.match(r'^[1-9]\d*\.\d{2}$', word)  is not None

def hasRealAndCurrency(word):
    return '$' in word and isRealNumber(word.split('$')[1])


def nature(word):
    flags = [isAlphabethic(word), isNumeric(word), isAlphaNumeric(word), isNumberWithDecimal(word), 
             isRealNumber(word), isCurrency(word), hasRealAndCurrency(word)]
    # none of the above
    mix = not any(flags)
    # none of the above and contains '$'
    mixc = mix and '$' in word
    
    return flags + [mix, mixc]

def get_word_binary_features(word, debug=False):
    if debug:
        features = {
            "isDate": isDate(word),
            "isZipCode": isZipCode(word),
            "isKnownCity": isKnownCity(word),
            "isKnownState": isKnownState(word),
            "isKnownCountry": isKnownCountry(word),
            "nature": nature(word)
        }
    else:
        features = [
            isDate(word),
            isZipCode(word),
            isKnownCity(word),
            isKnownState(word),
            isKnownCountry(word)] +  nature(word)
        features = [int(flag) for flag in features]
            
    return features

#### Numeric Features (neighbour distances from graph)
# -------------------------------------------------------------------------
def get_word_area_numeric_features(word_area, graph):
    """
    Fill the "undefined" values with the null distance (0)
    """
    edges = graph[word_area.idx]
    
    get_numeric = lambda x: x if x != UNDEFINED else 0
    
    rd_left   = get_numeric(edges["left"][1]) 
    rd_right  = get_numeric(edges["top"][1]) 
    rd_top    = get_numeric(edges["right"][1]) 
    rd_bottom = get_numeric(edges["bottom"][1]) 
    
    return [rd_left, rd_right, rd_top, rd_bottom]


### Run Feature Calculator Process
# -------------------------------------------------------------------------
def run_feature_calculator(normalized_dir, target_dir):
    print("Running feature calculator")
        
    image_filepaths, word_filepaths, _ = get_normalized_filepaths(normalized_dir)
    
    for image_filepath, word_filepath in zip(image_filepaths, word_filepaths):
        # read normalized data for one image
        image, word_areas, _ = load_normalized_example(image_filepath, word_filepath)

        # compute graph adj matrix for one image
        lines = line_formation(word_areas)
        image_width, image_height = cv_size(image)
        graph = graph_modeling(lines, word_areas, image_width, image_height)
        adj_matrix = form_adjacency_matrix(graph)
        
        # compute nodes feature matrix for one image
        word_binary_features = [get_word_binary_features(wa.content) for wa in word_areas]
        word_numeric_features = [get_word_area_numeric_features(wa, graph) for wa in word_areas]
        word_embeddings = [get_word_bpe(wa.content, bpemb_en) for wa in word_areas]

        # save node features and graph
        save_features(target_dir, image_filepath, word_embeddings, word_binary_features, word_numeric_features)
        save_graph(target_dir, image_filepath, graph, adj_matrix)

# -------------------------------------------------------------------------
# Run the process in batch mode
# -------------------------------------------------------------------------
def main():

    # configs for data storage
    PROJECT_ROOT_PREFIX = "/home/adrian/as/blogs/nanonets"

    NORMALIZED_PREFIX = "invoice-ie-with-gcn/data/normalized/"
    normalized_dir = os.path.join(PROJECT_ROOT_PREFIX, NORMALIZED_PREFIX)

    FEATURES_PREFIX = "invoice-ie-with-gcn/data/"
    features_dir = os.path.join(PROJECT_ROOT_PREFIX, FEATURES_PREFIX)

    run_feature_calculator(normalized_dir, features_dir)

if __name__ == "__main__":
    main()
