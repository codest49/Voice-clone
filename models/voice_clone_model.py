import torch.nn as nn

class VoiceCloneModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(VoiceCloneModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)  # (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, :, :])  # (batch_size, sequence_length, 
input_size)
        return out