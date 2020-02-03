# globals
from collections import namedtuple

# named tuples
WordArea = namedtuple('WordArea', 'left, top, right, bottom, content, idx')
TrainingParameters = namedtuple("TrainingParameters", "dataset model learning_rate epochs hidden1 num_hidden_layers dropout weight_decay early_stopping max_degree")

# trainig parameters
FLAGS = TrainingParameters('receipts', 'gcnx_cheby', 0.001, 200, 16, 2, 0.6, 5e-4, 10, 3)
UNDEFINED="undefined"
CLASSES = ["company", "date", "address", "total", UNDEFINED]

