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
             lw=lw, label='ROC曲线 (AUC = %0.4f)' % test_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳率')
    plt.ylabel('真阳率')
    hidden = "%0.0f%%" % ((frac_hidden - 0.35) * 100)
    title = 'ROC Curve:\nFB-{} graph, {} hidden, {}'.format(graph_name, hidden, 'gae')
    print(hidden)
    #plt.title(title)
    plt.legend(loc="lower right")
    TABLE_RESULTS_DIR = './results/tables/' + 'FB-{} graph, {} hidden, {}'.format(graph_name, hidden, 'gae') +'.png'
    plt.savefig(TABLE_RESULTS_DIR)
    plt.show()

#show_roc_curve('combined', 0.5, 'gae_edge_emb')

def draw_graph(dataset):
    graph = []
    name = 'txt'
    with open("{}/{}.{}".format(dataset, dataset, name), 'rb') as f:
        graph = nx.read_edgelist(f, create_using=nx.Graph(), nodetype=str, encoding='latin1', data=(('weight', float),))

   # nx.draw(graph)
    nx.draw(graph, pos=nx.random_layout(graph), graph='b', with_labels=True, font_size=7, node_size=200)
    plt.savefig("results/tables/{}.pdf".format(dataset))
    plt.show()

#draw_graph('hamster')

#FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
FB_EGO_USERS = [414]
for ego in FB_EGO_USERS:
    with open("facebook/{}.edges".format(ego), 'rb') as f:
        graph = nx.read_edgelist(f, create_using=nx.Graph(), nodetype=str, encoding='latin1', data=(('weight', float),))

    nx.draw(graph, pos=nx.random_layout(graph), graph='b', with_labels=True, width=0.3, font_size=7, node_size=200)
    plt.rcParams['figure.figsize'] = (8.0, 4.0)
    plt.savefig("results/tables/{}.png".format(ego))
    plt.show()