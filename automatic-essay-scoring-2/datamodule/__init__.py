from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class EssayDataset(Dataset):
    data: 'DataFrame'

    def __getitem__(self, index) -> 'Series':
        return self.data.iloc[index]
    
    def __len__(self):
        return len(self.data)

def collate_fn(batch, tokenizer, train):
    text = [item['full_text'] for item in batch]
    inputs = tokenizer(
        text,
        padding='longest',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )

    if train:
        labels = torch.tensor([item['score'] for item in batch], dtype=torch.long)
        inputs['labels'] = labels

    return inputs