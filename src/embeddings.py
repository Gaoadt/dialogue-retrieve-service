from src.dataset import *
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