import networkx as nx
import matplotlib.pyplot as plt
import link_prediction_scores as lp
import pickle

##1. Generate Bar Plots with Network Statistics
RANDOM_SEED = 0

# Open FB results
fb_results = None
with open('./results/fb-experiment-4-results.pkl', 'rb') as f:
    fb_results = pickle.load(f, encoding='latin1')

##2. Plot ROC Curves
# Plot ROC curve given graph_name, frac_hidden, and link prediction method (e.g. aa, for adamic-adar)
def show_roc_curve(graph_name, frac_hidden, method):
    results_dict = fb_results['fb-{}-{}-hidden'.format(graph_name, frac_hidden)]
    roc_curve = results_dict[method]['test_roc_curve']
    test_roc = results_dict[method]['test_roc']
    fpr, tpr, _ = roc_curve

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % test_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    hidden = "%0.0f%%" % ((frac_hidden - 0.35) * 100)
    title = 'ROC Curve:\nFB-{} graph, {} hidden, {}'.format(graph_name, hidden, 'gae')
    print(hidden)
    plt.title(title)
    plt.legend(loc="lower right")
    TABLE_RESULTS_DIR = './results/tables/' + 'FB-{} graph, {} hidden, {}'.format(graph_name, hidden, 'gae') +'.pdf'
    plt.savefig(TABLE_RESULTS_DIR)
    plt.show()

# Try it out!
show_roc_curve('combined', 0.5, 'gae_edge_emb')