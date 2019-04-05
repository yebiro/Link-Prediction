import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import link_prediction_scores as lp
import pickle, json
import os
import tensorflow as tf

# Runtime parameters
NUM_REPEATS = 1
RANDOM_SEED = 0
FRAC_EDGES_HIDDEN = [0.15, 0.3]

# Read in brightkite network
TRAIN_TEST_SPLITS_FOLDER = './train-test-splits/'
network_dir = './brightkite/brightkite-adj.pkl'
brightkite_adj = None
with open(network_dir, 'rb') as f:
    brightkite_adj = pickle.load(f)



### ---------- RUN LINK PREDICTION TESTS ---------- ###
for i in range(NUM_REPEATS):
    brightkite_results = {} # nested dictionary: experiment --> results

    # Check existing experiment results, increment file number by 1
    past_results = os.listdir('./results/')
    experiment_num = 0
    experiment_file_name = 'brightkite-experiment-{}-results.json'.format(experiment_num)
    while (experiment_file_name in past_results):
        experiment_num += 1
        experiment_file_name = 'brightkite-experiment-{}-results.json'.format(experiment_num)

    brightkite_results_dir = './results/' + experiment_file_name

    # Iterate over fractions of edges to hide
    for frac_hidden in FRAC_EDGES_HIDDEN:
        val_frac = 0.05
        test_frac = frac_hidden - val_frac

        # Read train-test split
        experiment_name = 'brightkite-{}-hidden'.format(frac_hidden)
        print("Current experiment: ", experiment_name)
        train_test_split_file = TRAIN_TEST_SPLITS_FOLDER + experiment_name + '.pkl'

        # Run all link prediction methods on current graph, store results
        brightkite_results[experiment_name] = lp.calculate_all_scores(brightkite_adj, features_matrix=None, 
                                                     directed=False, \
                                                     test_frac=test_frac, val_frac=val_frac, \
                                                     random_state=RANDOM_SEED, verbose=2,
                                                     train_test_split_file=train_test_split_file,
                                                     tf_dtype=tf.float16)

        # Save experiment results at each iteration
        with open(brightkite_results_dir, 'w') as fp:
            json.dump(brightkite_results, fp, indent=4)

    # Save final experiment results
    with open(brightkite_results_dir, 'w') as fp:
        json.dump(brightkite_results, fp, indent=4)