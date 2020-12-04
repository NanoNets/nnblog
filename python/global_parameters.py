# global parameters

from collections import namedtuple

# named tuples
WordArea = namedtuple('WordArea', 'left, top, right, bottom, content, idx')
TrainingParameters = namedtuple("TrainingParameters", "dataset model learning_rate epochs hidden1 num_hidden_layers dropout weight_decay early_stopping max_degree data_split")

# trainig parameters:
# --------------------------------------------------------------------------------
# dataset:        Selectes the dataset to run on
# model:          Defines the type of layer to be applied
# learning_rate:  Initial learning rate.
# epochs:         Number of epochs to train.
# hidden1:        Number of units in hidden layer 1.
# dropout:        Dropout rate (1 - keep probability).
# weight_decay:   Weight for L2 loss on embedding matrix.
# early_stopping: Tolerance for early stopping (# of epochs).
# max_degree:     Maximum Chebyshev polynomial degree.
FLAGS = TrainingParameters('receipts', 'gcnx_cheby', 0.001, 200, 16, 2, 0.6, 5e-4, 10, 3, [.4, .2, .4])

# output classes
UNDEFINED="undefined"
CLASSES = ["company", "date", "address", "total", UNDEFINED]

