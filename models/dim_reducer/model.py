import torch
import torch.nn as nn

class DimensionReducer:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        
    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Заглушка для тестирования
        return torch.randn(input_tensor.shape[0], input_tensor.shape[1] // 2)
