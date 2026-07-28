import torch
import torch.nn as nn
import torch.nn.functional as F

class SkinCareAttentionTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super(SkinCareAttentionTransformer, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(input_dim * d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(-1)
        embedded = F.relu(self.input_projection(x_seq))
        attn_out = self.transformer(embedded)
        flat = attn_out.reshape(attn_out.size(0), -1)
        return torch.sigmoid(self.head(flat))