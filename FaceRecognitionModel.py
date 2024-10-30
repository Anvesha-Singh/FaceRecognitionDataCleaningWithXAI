import torch
import torch.nn as nn

class FaceRecognitionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(FaceRecognitionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size  # Store input_size as an instance attribute
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Reshape the input to (batch_size, sequence_length, input_size)
        # Assuming each image is flattened to size input_size
        x = x.view(x.size(0), -1, self.input_size)  # Use self.input_size here
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output for classification
        return out