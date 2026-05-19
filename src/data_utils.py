import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import random
from pathlib import Path
from .audio_utils import read_audio_file
from .musan import Musan
from .rir import RIRReverberation

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"

LABEL_MAP = {
    "bonafide": 1,
    "spoof": 0,
}

def parse_trial_line(line):
    fields = line.strip().split()
    if len(fields) != 10:
        raise ValueError(f"Expected 10 fields in trial line, got {len(fields)}: {line.strip()}")

    spk_id, key, _, _, _, _, _, _, label, _ = fields
    if label not in LABEL_MAP:
        raise ValueError(f"Unsupported label '{label}' in trial line: {line.strip()}")

    return {
        "speaker_id": spk_id,
        "utterance_id": key,
        "label_name": label,
        "label": LABEL_MAP[label],
        "trial_line": line.rstrip("\n"),
    }


def load_trial_records(dir_meta):
    with open(dir_meta, "r") as file:
        return [parse_trial_line(line) for line in file if line.strip()]


def genSpoof_list(dir_meta):
    d_meta = {}
    file_list = []
    for record in load_trial_records(dir_meta):
        file_list.append(record["utterance_id"])
        d_meta[record["utterance_id"]] = record["label"]
    return d_meta, file_list
    """
    if is_train:
        for line in l_meta:
            _, key, _, _, _, _, _, _,  label, _ = line.strip().split("\t")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            _, key, _, _, _, _ = line.strip().split("\t")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, _, label = line.strip().split("\t")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list
    """

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        if x_len == max_len:
            return x
        stt = np.random.randint(0, x_len - max_len + 1)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class TrainDataset(Dataset):
    def __init__(
        self,
        list_IDs,
        labels,
        base_dir,
        add_noise=False,
        musan_path="musan_data",
        rir_path="RIR_data",
    ):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.add_noise = add_noise

        self.DA = {}
        self.category = ['noise','speech','music']
        if self.add_noise:
            self.DA['MUS'] = Musan(
                        str(musan_path)
                    )
            self.DA['RIR'] = RIRReverberation(
                        str(rir_path)
                    )
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = read_audio_file(self.base_dir / f"{key}.flac")
        if self.add_noise:
            if 0.5 > random.random():
                if random.randint(0, 1) == 0:
                    category = random.choice(self.category)
                    X = self.DA['MUS'](X, category)
                else:
                    X = self.DA['RIR'](X)    
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class TestDataset(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = read_audio_file(self.base_dir / f"{key}.flac", frames=self.cut)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


class LabeledEvalDataset(Dataset):
    def __init__(self, trial_records, base_dir, return_trial_line=False):
        self.trial_records = trial_records
        self.base_dir = base_dir
        self.return_trial_line = return_trial_line
        self.cut = 64600

    def __len__(self):
        return len(self.trial_records)

    def __getitem__(self, index):
        record = self.trial_records[index]
        key = record["utterance_id"]
        X, _ = read_audio_file(self.base_dir / f"{key}.flac", frames=self.cut)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)

        if self.return_trial_line:
            return x_inp, record["label"], key, record["trial_line"]

        return x_inp, record["label"], key
