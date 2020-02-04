# model runner
import warnings
import time
import os
import sys
from scipy.linalg import block_diag
from scipy import sparse
import tensorflow as tf

from global_parameters import UNDEFINED, CLASSES, FLAGS
from data_access import *
from gcn.utils import *
from gcn.models import GCNX, GCN, MLP

# -------------------------------------------------------------------------
# Data loader
# -------------------------------------------------------------------------
def onehot(idx, num_classes):
    """
    1-hot encoding. 
    """
    encoding = np.zeros(num_classes)
    encoding[1] = 1
    return encoding

def encode_labels(class_indices, num_classes):
    encodings = [onehot(idx, num_classes) for idx in class_indices]
    return np.array(encodings, dtype=np.float)

def form_batch(features_dir, class2idx):
    """
    Creates a single batch with all the graph instances from the training set. For that purpose, 
    it concatenates all the adjacency matrices into a (sparse) block-diagonal matrix 
    where each block corresponds to the adjacency matrix of one graph instance.
    The individual graph adj matrix can be of different sizes.
    
    Input: a list of M individual graph examples
        - graphs: (list of tuples) a list of length M, where each tuple contains 3 matrices: 
                    1. feature_matrix (N, 318): represents the nodes of the graph (words)
                    2. adjacency_matrix (N, N): represents the edgest of the graph
                    3. labels (N, 1): each element represents the category of a node (word tag)
    
    Output: a single graph containing M non-linkes independent sub-graphs
        - batch: (tuple) A single graph represented as a tuple, containing 3 matrices:
                    1. feature_matrix (M*N, 318) : concatenation of all the input feature matrices
                    2. adjacency_matrix (M*N, M*N): block-diagonal matrix
                    3. labels (M*N, 1): concatenation of all the input labels
    """
    X = []  # batch feature matrix
    A = []  # batch adjacency matrix
    CI = []  # batch class indices (labels)
    
    # maps each row in the output feature_matrix back to the individual graph where the row originated
    # this allows to pool the predicted node scores back into each independent graph
    example_ids = [] 
    
    feature_files, _, adj_matrix_files, entity_files = get_feature_filepaths(features_dir)
    
    for example_id, (f_file, am_file, e_file) in enumerate(zip(feature_files, adj_matrix_files, entity_files)):
        features, adj_matrix = read_features(f_file, am_file)
        entities = read_normalized_entities(e_file)
        X.append(features)
        A.append(adj_matrix)
        class_indices = np.asarray([[class2idx[entity]] for entity in entities]) 
        CI.append(class_indices)
        
        example_ids.append(example_id)

    X = np.vstack(X)
    A = block_diag(*A)
    CI = np.vstack(CI)
    
    features = sparse.lil_matrix(X)

    return features, A, CI, example_ids

def split_sets(class_indices, class2idx, split):
    """
    Splits the dataset according to a split distribution. 
    The split is done over the total number of nodes,
    regardless of which independent graph example the node belongs to.
    
    Although the nodes are "logically" split into 3 subsets (train, validation and test), 
    all the nodes are processed together by the model, so 3 full-sized arrays
    of labels are produced, one for each subset, where the labels 
    from the other subsets are masked (zeros).
    
    """
    # select actual labels (filter UNDEFINED)
    undefined_idx = class2idx[UNDEFINED]
    y_mask = np.array(class_indices != undefined_idx)
    y_idx = np.array([idx for idx, m in enumerate(y_mask) if m])
    
    # randomly select from the labeled nodes indices for each split set
    N = y_idx.shape[0]
    n_train, n_val = int(N*split[0]), int(N*split[1])
    np.random.shuffle(y_idx)
    idx_train = y_idx[:n_train]
    idx_val = y_idx[n_train:n_train + n_val]
    idx_test = y_idx[n_train + n_val:]

    # create a matrix of one-hot encoded labels
    labels = encode_labels(class_indices, len(class2idx))
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    print("\nLabels:")
    print("Number of Classes: {}".format(y_train.shape[1]))
    print("Number of Labeled Nodes: {}".format(N))
    print("Number of Training Nodes: {}".format(n_train))
    print("Number of Training Nodes per Class: {}".format(n_train//y_train.shape[1]))

    return y_train, y_val, y_test, train_mask, val_mask, test_mask

def data_loader(features_dir, class2idx, split):

    features, adj, class_indices, example_id =\
    form_batch(features_dir, class2idx)

    y_train, y_val, y_test, train_mask, val_mask, test_mask =\
    split_sets(class_indices, class2idx, split)
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



# -------------------------------------------------------------------------
# model runner
# -------------------------------------------------------------------------
def run_gcn_model(features_dir, class2idx, split):
    """
        Run semi-supervised learning process
    """    
    ## load dataset
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask =\
    data_loader(features_dir, class2idx, split)
    
    print("\nFeatures: {}".format(features.shape))
    print("Adjacency Matrix: {}".format(adj.shape))
    print("Labels: {}".format(y_train.shape))
    
    
    ## Run Semi-Supervised Training
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'gcnx_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCNX
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)


    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val, acc_val = [], []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        acc_val.append(acc)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping:
            if cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                print(" Validation cost is not improving. Early stopping...")
                break
            if 1.0 == round(np.mean(acc_val[-(FLAGS.early_stopping+1):-1]), 5):
                print("Validation accuracy reached 1.0. Early stopping...")
                break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# -------------------------------------------------------------------------
# Run the process in batch mode
# -------------------------------------------------------------------------
def main():
    print("tensorflow version:{}   -   GPU: {}".format(tf.__version__, tf.test.is_gpu_available()))

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # class dictionary
    class2idx = {c:idx for idx, c in enumerate(CLASSES)}
    
    # configs for data storage
    PROJECT_ROOT_PREFIX = os.path.abspath('../')
    NORMALIZED_PREFIX = "data/sroie2019/normalized/"
    normalized_dir = os.path.join(PROJECT_ROOT_PREFIX, NORMALIZED_PREFIX)
    FEATURES_PREFIX = "data/sroie2019/"
    features_dir = os.path.join(PROJECT_ROOT_PREFIX, FEATURES_PREFIX)

    print(FLAGS)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #warnings.filterwarnings("ignore", category = DeprecationWarning, module = "tensorflow")
        run_gcn_model(features_dir, class2idx, FLAGS.data_split)

if __name__ == "__main__":
    main()
