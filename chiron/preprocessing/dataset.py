import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, preprocessor):
        self.texts = texts
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize and convert text to indices
        tokenized_text = self.preprocessor.tokenize_text(text)
        indices = self.preprocessor.tokens_to_indices(tokenized_text)

        # Convert indices to tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        return indices_tensor
