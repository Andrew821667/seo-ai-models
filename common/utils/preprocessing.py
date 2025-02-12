import torch
from transformers import AutoTokenizer
from typing import List, Dict, Union

class TextPreprocessor:
    """Класс для предобработки текстовых данных"""
    def __init__(
        self,
        max_length: int = 512,
        model_name: str = "bert-base-uncased"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def preprocess(self, text_data: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Предобработка текстовых данных"""
        if isinstance(text_data, str):
            text_data = [text_data]
            
        encoded = self.tokenizer(
            text_data,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return encoded
