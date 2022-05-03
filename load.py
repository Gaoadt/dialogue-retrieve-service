import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse

from tqdm import tqdm
import typing as tp
import numpy.typing as ntp

import json
from pathlib import Path


def load_multiwoz(split, path='./data/multiwoz/data/MultiWOZ_2.2', order=None):
    data_dir = Path(path) / split
    data = []
    data_parts = list(data_dir.iterdir())
    if order:
        data_parts = [data_dir / order_item for order_item in order]
    for data_part in tqdm(data_parts):
        print(data_part)
        with data_part.open() as f:
            data.extend(json.load(f))
    return data


from itertools import accumulate


class Utterance:
    def __init__(self, utterance: str, speaker: str, turn_id: str, **meta: tp.Any):
        self.utterance = utterance
        self.speaker = speaker
        self.turn_id = turn_id
        self.meta = meta

    def __str__(self) -> str:
        return self.utterance

    def __repr__(self) -> str:
        return f"[{self.turn_id:>2}] {self.speaker:>8}: \"{self.utterance}\""

    @classmethod
    def from_multiwoz_v22(cls, utterance: tp.Dict[str, tp.Any]) -> 'Utterance':
        return cls(**utterance)


class Dialogue:
    def __init__(self, utterances: tp.List[Utterance], dialogue_id: str, **meta: tp.Any):
        self.utterances = utterances
        self.dialogue_id = dialogue_id
        self.meta = meta

    def __len__(self) -> int:
        return len(self.utterances)

    def __str__(self) -> str:
        return "\n".join(str(utt) for utt in self.utterances)

    def __repr__(self) -> str:
        return f"[{self.dialogue_id}]\n" + '\n'.join(repr(utt) for utt in self.utterances)

    def __getitem__(self, i) -> Utterance:
        return self.utterances[i]

    def __iter__(self) -> tp.Iterator[Utterance]:
        return iter(self.utterances)

    @classmethod
    def from_multiwoz_v22(cls, dialogue: tp.Dict[str, tp.Any]) -> 'Dialogue':
        utterances = [Utterance.from_multiwoz_v22(utt) for utt in dialogue['turns']]
        dialogue_id = dialogue['dialogue_id']
        meta = {key: val for key, val in dialogue.items() if key not in ['turns', 'dialogue_id']}
        return cls(utterances=utterances, dialogue_id=dialogue_id, **meta)


class DialogueDataset(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.utterances = [utt.utterance for dialog in self for utt in dialog]

        self._dialogue_start = list(accumulate([0] + [len(dialogue) for dialogue in self]))

        self._utt_dialogue_id = [0] * len(self.utterances)
        self._utt_id = [0] * len(self.utterances)
        for d_start in self._dialogue_start[1:-1]:
            self._utt_dialogue_id[d_start] = 1
        current_utt_id = 0
        for i in range(len(self._utt_id)):
            if self._utt_dialogue_id[i] == 1:
                current_utt_id = 0
            self._utt_id[i] = current_utt_id
            current_utt_id += 1
        self._utt_dialogue_id = list(accumulate(self._utt_dialogue_id))

        self._dial_id_mapping = {dialogue.dialogue_id: i
                                 for i, dialogue in enumerate(self)}

    def get_dialogue_by_idx(self, idx: int) -> Dialogue:
        udi = self._utt_dialogue_id[idx]
        return self[udi]

    def get_utterance_by_idx(self, idx: int) -> Utterance:
        udi = self._utt_dialogue_id[idx]
        ui = self._utt_id[idx]
        return self[udi][ui]

    def get_dialog_start_idx(self, dialogue: 'Dialog') -> int:
        dialogue_idx = self._dial_id_mapping[dialogue.dialogue_id]
        d_start = self._dialogue_start[dialogue_idx]
        return d_start

    @classmethod
    def from_miltiwoz_v22(cls, multiwoz_v22: tp.List[tp.Dict[str, tp.Any]]) -> 'DialogueDataset':
        dialogues = [Dialogue.from_multiwoz_v22(dialog) for dialog in multiwoz_v22]
        return cls(dialogues)


class Subset(DialogueDataset):
    def __init__(self, dialogues: DialogueDataset, subset: tp.Iterable):
        subset_dialogues = [dialogues[idx] for idx in subset]
        super().__init__(subset_dialogues)


test = DialogueDataset.from_miltiwoz_v22(load_multiwoz('test'))
val = DialogueDataset.from_miltiwoz_v22(load_multiwoz('dev'))
train = DialogueDataset.from_miltiwoz_v22(load_multiwoz('train', order=['dialogues_001.json', 'dialogues_011.json',
                                                                        'dialogues_007.json', 'dialogues_010.json',
                                                                        'dialogues_017.json', 'dialogues_005.json',
                                                                        'dialogues_015.json', 'dialogues_012.json',
                                                                        'dialogues_016.json', 'dialogues_013.json',
                                                                        'dialogues_004.json', 'dialogues_009.json',
                                                                        'dialogues_003.json', 'dialogues_006.json',
                                                                        'dialogues_008.json', 'dialogues_002.json',
                                                                        'dialogues_014.json']))

from abc import ABC, abstractmethod
from tqdm.notebook import tqdm


class OneViewEmbedder(ABC):
    def __init__(self, config: tp.Optional[tp.Any] = None):
        self.config = config

    @abstractmethod
    def encode_dialogue(self, dialogue: Dialogue):
        return np.zeros((len(dialogue), 1), dtype=np.int32)

    def encode_dataset(self, dialogues: DialogueDataset):
        return np.concatenate([self.encode_dialogue(dialogue) for dialogue in tqdm(dialogues)], axis=0)

    def encode_new_dialogue(self, dialogue: Dialogue):
        return self.encode_dialogue(dialogue)

    def encode_new_dataset(self, dialogues: DialogueDataset):
        return np.concatenate([self.encode_new_dialogue(dialogue) for dialogue in tadm(dialogues)], axis=0)


class CachedEmbeddings():
    def __init__(self, dialogues: DialogueDataset, embeddings: np.array,
                 test_dialogues: DialogueDataset = None, test_embeddings: np.array = None):
        self.dialogues = dialogues
        self.embeddings = embeddings
        self.test_dialogues = test_dialogues
        self.test_embeddings = test_embeddings

    def encode_dialogue(self, dialogue: Dialogue) -> np.array:
        idx = self.dialogues.get_dialog_start_idx(dialogue)
        return self.embeddings[idx:idx + len(dialogue)]

    def encode_new_dialogue(self, dialogue: Dialogue):
        idx = self.test_dialogues.get_dialog_start_idx(dialogue)
        return self.test_embeddings[idx:idx + len(dialogue)]

    def encode_utterances(self, utts):
        return self.embeddings[utts]


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


from collections import defaultdict


class FrequencyDialogueGraph:
    def __init__(self, dialogues: DialogueDataset, embedder: OneViewEmbedder, clustering: OneViewClustering):
        self.dialogues: DialogueDataset = dialogues
        self.clustering: OneViewClustering = clustering
        self.embedder: OneViewEmbedder = embedder

        self.n_vertices = clustering.get_nclusters() + 1
        self.start_node = self.n_vertices - 1

        self.edges = [[0] * self.n_vertices for _ in range(self.n_vertices)]

        self.eps = 1e-5

    #         self._build()

    def _add_dialogue(self, dialogue: Dialogue) -> None:
        utt_idx = self.dialogues.get_dialog_start_idx(dialogue)
        current_node = self.start_node
        for utt in dialogue:
            next_node = self.clustering.get_utterance_cluster(utt_idx).id
            self.edges[current_node][next_node] += 1
            current_node = next_node
            utt_idx += 1

    def _build(self):
        for dialogue in self.dialogues:
            self._add_dialogue(dialogue)

        self.probabilities = [np.array(self.edges[v]) / np.sum(self.edges[v])
                              for v in range(self.n_vertices)]

    def _dialogue_success_rate(self, dialogue: Dialogue, acc_ks=None) -> float:
        if acc_ks is None:
            acc_ks = []
        acc_ks = np.array(acc_ks)

        d_embs = self.embedder.encode_new_dialogue(dialogue)

        logprob = 0
        accuracies = np.zeros(len(acc_ks))

        current_node = self.start_node
        for utt, emb in zip(dialogue, d_embs):
            next_node = self.clustering.predict_cluster(emb, utt, dialogue).id
            prob = self.probabilities[current_node][next_node]
            prob = max(prob, self.eps)
            logprob -= np.log(prob) * prob

            next_cluster_ind = (self.probabilities[current_node] >= prob).sum()
            accuracies = accuracies + (next_cluster_ind <= acc_ks)

            current_node = next_node
        accuracies /= len(dialogue)
        return logprob, accuracies

    def success_rate(self, test: DialogueDataset, acc_ks=None):
        if acc_ks is None:
            acc_ks = []
        logprob = 0
        accuracies = np.zeros(len(acc_ks))
        for dialogue in test:
            lp, acc = self._dialogue_success_rate(dialogue, acc_ks)
            logprob += lp
            accuracies += acc
        logprob /= len(test)
        accuracies /= len(test)
        return logprob, accuracies


class RetrieveService:
    def __init__(self, train, embedder, embeddings, cluster_count, response_emb):
        self.algo = MiniBatchKMeansClustering
        self.embeddings = embeddings
        self.embedder = embedder
        self.cluster_count = cluster_count
        self.clustering = self.algo(n_clusters=cluster_count).fit(self.embeddings)
        self.clustering.clustering.labels_ = self.clustering.clustering.predict(self.embedder.embeddings)
        self.graph = FrequencyDialogueGraph(train, self.embedder, self.clustering)
        self.graph._build()
        self.train = train
        self.response_emb = response_emb

        self.db = [[] for i in range(cluster_count)]
        for i in range((len(response_emb))):
            self.db[self.clustering.clustering.predict(np.array([response_emb[i]]))[0]].append(i)

    def next_cluster(self, context_embedding):
        arg = np.array([context_embedding.tolist()], dtype=np.float32)
        current_cluster = self.clustering.clustering.predict(arg)[0]
        return self.graph.probabilities[current_cluster].argmax()

    def retrieve(self, context_embedding):
        next = self.next_cluster(context_embedding)
        emb = self.response_emb[self.db[next]]
        return self.train.utterances[self.db[next][context_embedding.dot(emb.T).argmax()]]

# Loading convert
test_convert = np.load("data/embeddings/test_convert_context.np.npy")
resp_test_convert = np.load("data/embeddings/test_convert_responses.np.npy")

train_convert = np.load("data/embeddings/train_convert_context.np.npy")
resp_train_convert = np.load("data/embeddings/train_convert_responses.np.npy")

cached_convert_context = CachedEmbeddings(train, train_convert, test, test_convert)
cached_convert_response = CachedEmbeddings(train, resp_train_convert, test, resp_test_convert)

def build_service(cluster_count = 100):
    service = RetrieveService(train, cached_convert_context, train_convert, cluster_count, resp_train_convert)
    return service
