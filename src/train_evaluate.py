import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

print(f"Current Working Directory: {os.getcwd()}")

from vanilla_transformer import TimeSeriesTransformer
#from vanilla_transformer import PositionalEncoding

def create_sequences(data, targets, look_back=24*60, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:(i + look_back)])
        y.append(targets[i + look_back:i + look_back + forecast_horizon])
    return np.array(X), np.array(y)

def train(model, train_loader, test_loader, criterion, optimizer, scheduler):
    # Training parameters
    num_epochs = 15
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Check if a checkpoint exists and load it if available
    checkpoint_path = 'checkpoint.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for batch_in, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)

            # Adjust squeeze as necessary
            loss = criterion(outputs, batch_y.squeeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (batch_in + 1) % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch_in+1}: Train Loss: {train_loss / (batch_in + 1):.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_in, (batch_X, batch_y) in enumerate(test_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.squeeze(1)).item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Checkpoint saving and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break


def main():
    # Load Dataset
    df = pd.read_csv("/home/bisnu-sarkar/Deep_learning_projects/Electric-Power-Consumption/data/preprocessed_data.csv")
    df = df.set_index('Datetime')
    X = df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)
    y = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]

    print(type(X), X.shape, X.head(1))

    print("Loaded")
    # Scale features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_scaled = X_scaled.astype(np.float32)
    y_scaled = y_scaled.astype(np.float32)

    # Create sequences
    look_back = 24 * 60
    forecast_horizon = 1
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, look_back, forecast_horizon)

    # Train-test split
    test_size = int(0.25 * len(X_seq))
    X_train, X_test = X_seq[:-test_size], X_seq[-test_size:]
    y_train, y_test = y_seq[:-test_size], y_seq[-test_size:]

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create DataLoaders
    batch_size = 1
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[2]
    model = TimeSeriesTransformer(input_dim, output_dim)
    model.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    train(model, train_loader, test_loader, criterion, optimizer, scheduler)

if __name__ == "__main__":
    main()
