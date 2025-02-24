import torch
from torch.utils.data import Dataset, DataLoader

class VoiceCloneDataset(Dataset):
    def __init__(self, features_path):
        self.features = np.load(features_path)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx])

def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for inputs in train_loader:
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, inputs)  # Calculate loss
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")