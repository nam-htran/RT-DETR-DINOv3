import os
import math
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch import Tensor

class HuggingFaceTeacherWrapper(nn.Module):
    def __init__(self, model_id: str, token: str = None):
        super().__init__()
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Loading teacher model '{model_id}' from Hugging Face...")
        config = AutoConfig.from_pretrained(model_id, token=token)
        self._model = AutoModel.from_pretrained(model_id, token=token)
        self.is_vit = "vit" in config.model_type.lower()
        self._feature_dim = (
            self._model.config.hidden_size
            if self.is_vit
            else self._model.config.hidden_sizes[-1]
        )
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Detected {'ViT' if self.is_vit else 'ConvNeXT'} architecture. Feature dim: {self._feature_dim}")

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._model(pixel_values=x, output_hidden_states=True)
        if self.is_vit:
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            b, s, d = patch_tokens.shape
            h = w = int(math.sqrt(s))
            return patch_tokens.permute(0, 2, 1).reshape(b, d, h, w)
        return outputs.hidden_states[-1]