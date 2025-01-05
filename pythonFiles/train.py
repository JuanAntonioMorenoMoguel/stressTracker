import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to log messages to a file
def log_message(message, log_file_path="trans_log.txt"):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')

log_message(f"Using device: {device}")

# Load the dataset
file_path = '/home/seshasaianeeshteja_vempa_student_uml_edu/IOT_Final/nurse_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)
data = data.drop(columns=['datetime', 'id'])

# Data exploration and preprocessing
data = data.dropna()
target_column = 'label'
feature_columns = [col for col in data.columns if col != target_column]

label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

X = data[feature_columns].values
y = data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_message(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim, num_heads, ff_dim, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)  # Residual connection
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)  # Residual connection
        x = self.global_pool(x.transpose(1, 2)).squeeze(2)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# Hyperparameters
input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)
embed_dim = 64
num_heads = 64
ff_dim = 1024
dropout_rate = 0.1

model = TransformerModel(input_dim, num_classes, embed_dim, num_heads, ff_dim, dropout_rate).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop with Best Model Saving
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10, model_save_path="best_transformer_model.pth"):
    model.train()
    best_accuracy = 0.0  # Track the best accuracy
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0

        # Training phase
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Calculate training accuracy
        train_accuracy = correct / total

        # Validation phase
        val_accuracy = evaluate_model(model, test_loader, device)

        # Save the model if it achieves the best accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            log_message(f"New best model saved with accuracy: {best_accuracy:.4f}")

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        log_message(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Time: {epoch_time:.2f} sec")

# Evaluation Function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Train the model and save the best one
train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=200)

# Evaluate the final model
final_accuracy = evaluate_model(model, test_loader, device)
log_message(f"Final Test Accuracy: {final_accuracy:.4f}")