from tp2_aux import images_as_matrix, report_clusters, report_clusters_hierarchical
from metrics import calculate_metrics
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, DBSCAN
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import Normalizer, StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


N_FEATURES_EXTRACT = 6
N_FEATURES_SELECT = 6
K_RANGE = range(2, 9)
D_RANGE = np.arange(0.1, 6.1, 0.1)

data = images_as_matrix()
id_to_label = np.loadtxt("labels.txt", delimiter=",")
true_labels = id_to_label[:,1]
labeled = true_labels > 0
true_labels = true_labels[labeled]

def extract_features(method):
    method.fit(data)
    return method.transform(data)

extraction_methods = (
    PCA(n_components=N_FEATURES_EXTRACT), 
    KernelPCA(kernel="rbf", n_components=N_FEATURES_EXTRACT), 
    Isomap(n_components=N_FEATURES_EXTRACT)
    )
extraction_method_names = ["PCA", "kPCA", "Isomap"]

print("Extracting Features")
start = datetime.now()
extracted = [extract_features(method) for method in extraction_methods]
print(f"Completed in {datetime.now() - start}")
print(f"Total extracted features: {sum(m.shape[1] for m in extracted)}")
print("First sample:")
for method, m in zip(extraction_methods, extracted):
    print(f"\t{method.__class__.__name__}")
    print(f"\t{m[0,:]}")
extracted = np.hstack(extracted)

print("Selecting best features")
selector = SelectKBest(f_classif, k = N_FEATURES_SELECT).fit(extracted[labeled], true_labels)
print(selector.get_feature_names_out(
        [f"{method}{i}" for i in range(N_FEATURES_EXTRACT) for method in extraction_method_names]
    )
)
extracted = selector.transform(extracted)

extracted = Normalizer().fit_transform(extracted)

clustering_methods = (
    ("Agglomerative", lambda k : AgglomerativeClustering(n_clusters=k)),
    ("Spectral", lambda k : SpectralClustering(assign_labels="cluster_qr", n_clusters=k)),
    ("KMeans", lambda k : KMeans(n_clusters=k))
)

def plot_metrics(clustering_methods, varying_parameter, parameter_name):
    purity_plot = []
    precision_plot = []
    recall_plot = []
    f1_plot = []
    rand_plot = []
    kmeans_loss_plot = []
    all_clusterings = {}
    for method_name, method in clustering_methods:
        purity_plot.append((method_name, []))
        precision_plot.append((method_name, []))
        recall_plot.append((method_name, []))
        f1_plot.append((method_name, []))
        rand_plot.append((method_name, []))
        all_clusterings[method_name] = []
        if method_name == "KMeans":
            kmeans_loss_plot.append((method_name, []))
        for x in varying_parameter:
            method_obj = method(x)
            pred = method_obj.fit_predict(extracted)[labeled]
            all_clusterings[method_name].append(pred)
            purity, precision, recall, f1, rand = calculate_metrics(pred, true_labels)
            purity_plot[-1][1].append(purity)
            precision_plot[-1][1].append(precision)
            recall_plot[-1][1].append(recall)
            f1_plot[-1][1].append(f1)
            rand_plot[-1][1].append(rand)
            if method_name == "KMeans":
                kmeans_loss_plot[-1][1].append(method_obj.inertia_)
    fig, ax = plt.subplots(2, 3, sharex=True)
    
    def plot_metric(i, metric_name, plots):
        legend = []
        for method_name, value in plots:
            legend.append(ax.flat[i].plot(varying_parameter, value, label=method_name)[0])
        ax.flat[i].legend(handles=legend)
        ax.flat[i].set(xlabel = parameter_name)
        ax.flat[i].set_title(metric_name)
    fig.tight_layout()
    plot_metric(0, "purity", purity_plot)
    plot_metric(1, "precision", precision_plot)
    plot_metric(2, "recall", recall_plot)
    plot_metric(3, "f1", f1_plot)
    plot_metric(4,"rand index", rand_plot)
    if len(kmeans_loss_plot) > 0:
        plot_metric(5, "kmeans loss", kmeans_loss_plot)
    return all_clusterings

all_clusterings = plot_metrics(clustering_methods, K_RANGE, "k clusters")
aglomerative_distance = (
    ("Agglomerative", lambda d : AgglomerativeClustering(n_clusters=None, distance_threshold=d)),
    ("DBSCAN", lambda d : DBSCAN(eps=d))
)
plot_metrics(aglomerative_distance, D_RANGE, "distance")

def report(name, x):
    report_clusters(id_to_label[labeled][:,0], all_clusterings[name][x-K_RANGE.start], f"{name}-{x}.html")

report("Agglomerative", 3)
report("Spectral", 3)
report("KMeans", 3)
plt.show()
