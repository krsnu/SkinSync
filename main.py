import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerSkincareRecommender(nn.Module):
    def __init__(self, num_conditions=5, num_ingredients=8, d_model=32, nhead=4, num_layers=2):
        super(TransformerSkincareRecommender, self).__init__()
        
        self.input_projection = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=64, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(num_conditions * d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_ingredients),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        
        embedded = F.relu(self.input_projection(x))
        
        transformer_out = self.transformer_encoder(embedded)
        
        flattened = transformer_out.reshape(transformer_out.size(0), -1)
        ingredient_scores = self.fc_out(flattened)
        
        return ingredient_scores

sample_user_vector = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)

model = TransformerSkincareRecommender()
scores = model(sample_user_vector)

print("Transformer Attention Model Output Shapes:", scores.shape)
print("Ingredient Scores (Un-trained):", scores.detach().numpy())