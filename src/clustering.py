from abc import ABC, abstractmethod
from src.dataset import *


class Cluster:
    def __init__(self, cluster_id, utterances):
        self.id = cluster_id
        self.utterances = utterances

    def __getitem__(self, idx):
        return self.utterances[idx]

    def __iter__(self):
        return iter(self.utterances)

    def __len__(self):
        return len(self.utterances)


class OneViewClustering(ABC):
    def __init__(self):
        self.size = 1

    @abstractmethod
    def fit(embeddings: np.array) -> 'OneViewClustering':
        self.size = embeddings.shape[0]
        self.cluster = Cluster(0, np.arange(self.size))
        return self

    @abstractmethod
    def get_cluster(self, idx: int) -> Cluster:
        assert idx == 1
        return self.cluster[0]

    @abstractmethod
    def get_utterance_cluster(self, utt_idx: int) -> Cluster:
        return self.cluster[0]

    @abstractmethod
    def get_nclusters(self) -> int:
        return 1

    @abstractmethod
    def predict_cluster(self, embedding: np.array,
                        utterance: tp.Optional[Utterance] = None,
                        dialogue: tp.Optional[Dialogue] = None):
        return self.cluster[0]

    @abstractmethod
    def get_labels(self) -> np.array:
        return np.zeros(self.size)


from collections import defaultdict


class SklearnClustering(OneViewClustering):
    def __init__(self, clustering, **config):
        self.clustering = clustering(**config)
        self.fitted = False

    def fit(self, embeddings: np.array) -> 'SklearnClustering':
        self.clustering.fit(embeddings)

        self.clusters = defaultdict(list)
        for idx, cluster in enumerate(self.clustering.labels_):
            self.clusters[cluster].append(idx)
        for key in self.clusters:
            self.clusters[key] = Cluster(key, np.array(self.clusters[key]))

        self.fitted = True
        return self

    def get_cluster(self, idx) -> Cluster:
        assert self.fitted, "SklearnClustering must be fitted"
        return self.clusters[idx]

    def get_utterance_cluster(self, utterance_idx) -> Cluster:
        assert self.fitted, "SklearnClustering must be fitted"
        return self.clusters[self.clustering.labels_[utterance_idx]]

    def get_nclusters(self) -> int:
        return self.clustering.n_clusters_

    def predict_cluster(self, embedding: np.array,
                        utterance: tp.Optional[Utterance] = None,
                        dialogue: tp.Optional[Dialogue] = None):
        labels = self.clustering.predict(embedding[None, :])
        return self.get_cluster(labels[0])

    def get_labels(self) -> np.array:
        return np.array([self.clusters[l].id for l in self.clustering.labels_])


from sklearn.cluster import KMeans


class KMeansClustering(SklearnClustering):
    def __init__(self, n_clusters=15, random_state=42, **config):
        self.n_clusters = n_clusters
        super().__init__(KMeans, n_clusters=n_clusters, random_state=random_state, **config)

    def get_nclusters(self) -> int:
        return self.n_clusters


from sklearn.cluster import KMeans, MiniBatchKMeans


class MiniBatchKMeansClustering(SklearnClustering):
    def __init__(self, n_clusters=15, random_state=42, **config):
        self.n_clusters = n_clusters
        super().__init__(MiniBatchKMeans, n_clusters=n_clusters, random_state=random_state, **config)

    def get_nclusters(self) -> int:
        return self.n_clusters