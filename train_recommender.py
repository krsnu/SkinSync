import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import joblib
from transformer_recommender import SkinCareAttentionTransformer

print("Loading Skincare Treatment Dataset")
df = pd.read_csv("Skincare_Treatment_Dataset.csv")

df['Ingredients_List'] = df['Ingrdients'].apply(
    lambda x: [ing.strip() for ing in str(x).split('+')]
)
feature_cols = ['Age_Group', 'Skin_Type', 'Skin_Subtype', 'Sensitivity', 'Concern', 'Internal_Type']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(df[feature_cols])
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(df['Ingredients_List'])

X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.float32)
class SkincareDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(SkincareDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

input_dim = X_encoded.shape[1]
output_dim = y_encoded.shape[1]
print(f" Training Transformer from scratch on {len(df)} samples...")
model = SkinCareAttentionTransformer(input_dim=input_dim, output_dim=output_dim)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
model.train()
for epoch in range(50):
    running_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

print("Training Complete!")

torch.save(model.state_dict(), "transformer_recommender.pth")
joblib.dump(encoder, "feature_encoder.pkl")
joblib.dump(mlb, "ingredient_binarizer.pkl")
print("Saved weights to 'transformer_recommender.pth' and encoding artifacts.")