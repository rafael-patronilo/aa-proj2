from tp2_aux import images_as_matrix, report_clusters, report_clusters_hierarchical
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import numpy as np

N_FEATURES = 6

data = images_as_matrix()

def pick_features(method):
    method.fit(data)
    return method.transform(data)

methods = PCA(n_components=N_FEATURES), KernelPCA(kernel="rbf", n_components=N_FEATURES), Isomap(n_components=N_FEATURES)
picked = np.hstack([pick_features(method) for method in methods])
print(picked.shape)
print(picked[0,:])