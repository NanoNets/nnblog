# global parameters


from collections import namedtuple

# named tuples
WordArea = namedtuple('WordArea', 'left, top, right, bottom, content, idx')
TrainingParameters = namedtuple("TrainingParameters", "dataset model learning_rate epochs hidden1 num_hidden_layers dropout weight_decay early_stopping max_degree data_split")

# trainig parameters
# flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
# flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
# flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
# flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
# flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
# flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
FLAGS = TrainingParameters('receipts', 'gcnx_cheby', 0.001, 200, 16, 2, 0.6, 5e-4, 10, 3, [.4, .2, .4])

# output classes
UNDEFINED="undefined"
CLASSES = ["company", "date", "address", "total", UNDEFINED]

