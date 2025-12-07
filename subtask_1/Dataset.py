import torch
from torch.utils.data import Dataset


class VADataset(Dataset):
    '''
    A PyTorch Dataset for Valence–Arousal regression.

    - Treats aspect and text as two separate segments.
    - Tokenizes the input using a HuggingFace tokenizer with two text inputs.
    - Compatible with BERT, mBERT, RoBERTa, XML-RoBERTa, DeBERTa, and other transformer models.
    - Returns:
        * input_ids: token IDs, shape [max_len]
        * attention_mask: mask, shape [max_len]
        * token_type_ids: segment IDs (if available, e.g., for BERT/mBERT/DeBERTa), shape [max_len]
        * labels: [Valence, Arousal], shape [2], float tensor

    Args:
        dataframe (pd.DataFrame): must contain "Text", "Aspect", "Valence", "Arousal".
        tokenizer: HuggingFace tokenizer (BERT, mBERT, RoBERTa, XML-RoBERTa, DeBERTa, etc.).
        max_len (int): max sequence length.
    '''
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    # def __getitem__(self, idx):
    #     text = f"{self.aspects[idx]}: {self.sentences[idx]}"
    #     encoded = self.tokenizer(
    #         text,
    #         truncation=True,
    #         padding="max_length",
    #         max_length=self.max_len,
    #         return_tensors="pt"
    #     )
    #     return {
    #         "input_ids": encoded["input_ids"].squeeze(0),
    #         "attention_mask": encoded["attention_mask"].squeeze(0),
    #         "labels": torch.tensor(self.labels[idx], dtype=torch.float)
    #     }

    def __getitem__(self, idx):
        # 将 aspect 和 text 作为两个独立的段输入
        aspect_text = self.aspects[idx]
        sentence_text = self.sentences[idx]
        
        # 使用两个文本输入，适用于 BERT、mBERT、RoBERTa、XML-RoBERTa、DeBERTa 等模型
        # 这些模型的 tokenizer 都支持两个文本段作为输入
        encoded = self.tokenizer(
            aspect_text,
            sentence_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
        
        # 如果 tokenizer 返回 token_type_ids，则添加到结果中
        # BERT/mBERT/DeBERTa 会返回 token_type_ids 来区分两个段
        # RoBERTa/XML-RoBERTa 可能不返回或返回全0，但模型仍能正常工作
        if "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        
        return result

