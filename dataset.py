import os
import torch
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
from tqdm import tqdm
import json


class TextDataset(Dataset):

    def __init__(self, data_file, tokenizer_path, train = True, max_length = 256):
        
        self.sp_model = SentencePieceProcessor(model_file=tokenizer_path)
        
        with open(data_file, "r") as file:
            self.data = json.load(file)
            
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
            self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    def text2ids(self, texts):
        return self.sp_model.encode(texts)

    def ids2text(self, ids):
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        encoded = self.data[str(item)]['tokens']
        if len(encoded) + 2 > self.max_length:
            encoded = encoded[:self.max_length - 2]
        indices = torch.full((self.max_length, ), self.pad_id, dtype=torch.int64)
        indices[0] = self.bos_id
        indices[1:len(encoded) + 1] = torch.tensor(encoded)
        indices[len(encoded) + 1] = self.eos_id
        return indices, len(encoded)+2