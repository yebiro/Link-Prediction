import networkx as nx
import pandas as pd
import pickle
import numpy as np
from gae.preprocessing import mask_test_edges_directed, mask_test_edges

RANDOM_SEED = 0

gplus_adj = pickle.load(open('./gplus/gplus-adj.pkl', 'rb'))

FRAC_EDGES_HIDDEN = [0.15, 0.3]
TRAIN_TEST_SPLIT_DIR = './train-test-splits/'

# Generate 1 train/test split for each frac_edges_hidden setting
for frac_hidden in FRAC_EDGES_HIDDEN:
    val_frac = 0.05
    test_frac = frac_hidden - val_frac

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Generate train_test_split: 
        # (adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false)
    train_test_split = mask_test_edges_directed(gplus_adj,
        test_frac=test_frac, val_frac=val_frac, 
        verbose=True, prevent_disconnect=False, false_edge_sampling='random')

    # Save split
    file_name = TRAIN_TEST_SPLIT_DIR + 'gplus-{}-hidden.pkl'.format(frac_hidden)
    with open(file_name, 'wb') as f:
        pickle.dump(train_test_split, f, protocol=2)
