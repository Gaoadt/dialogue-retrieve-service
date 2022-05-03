from src.dataset import *
from src.clustering import *
from src.embeddings import *

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

