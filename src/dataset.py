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

from itertools import accumulate


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


def load_dataset():
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
    return test, val, train
