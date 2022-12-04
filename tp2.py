from tp2_aux import images_as_matrix, report_clusters, report_clusters_hierarchical
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, rand_score
from sklearn.preprocessing import Normalizer
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

N_FEATURES = 6
K_RANGE = range(2, 9)

data = images_as_matrix()

def extract_features(method):
    method.fit(data)
    return method.transform(data)

extraction_methods = (
    PCA(n_components=N_FEATURES), 
    KernelPCA(kernel="rbf", n_components=N_FEATURES), 
    Isomap(n_components=N_FEATURES)
    )

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

extracted = Normalizer().fit_transform(extracted)

clustering_methods = (
    lambda k : AgglomerativeClustering(n_clusters=k),
    lambda k : SpectralClustering(assign_labels="cluster_qr", n_clusters=k),
    lambda k : KMeans(n_clusters=k)
)

clusterings = (
    clustering(k).fit_predict(data) for clustering in clustering_methods for k in K_RANGE
)

true_labels = None

def plot_metric(name, metric):
    for pred in clusterings:
        plt.plot(K_RANGE, [metric(true_labels, labels) for labels in pred])

for metric in [precision_score, recall_score, f1_score, rand_score]:
    plot_metric("", metric)