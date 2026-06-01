from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def train(
        model: nn.Module, 
        dataset: Dataset, 
        num_epochs: int, 
        batch_size: int, 
        learning_rate: float, 
        val_size: float,
        device: torch.device = "cpu"):
    # Create dataloaders
    train_size = int(len(dataset) * (1 - val_size))
    val_size = len(dataset) - train_size
    training_set, validation_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(training_set, batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size, shuffle=False)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    
    # Define loss
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.to(device)
    epochs_it = tqdm(range(num_epochs), desc="Epochs", leave=True)
    for i in epochs_it:
        model.train()
        
        train_batches_it = tqdm(train_loader, desc="Training", leave=False)
        for x, y in train_batches_it:
            x = x.to(device)
            y = y.to(device)

            out = model(x)[0]
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            accuracy = (torch.argmax(out, dim=1) == y).to(torch.float32).mean()
            
            train_batches_it.set_postfix_str(f"cross_entropy_loss: {loss.item():.4f}, accuracy: {accuracy.item():.4f}")

        with torch.no_grad():
            model.eval()
            loss = 0
            accuracy = 0
            val_batches_it = tqdm(val_loader, desc="Validation", leave=False)
            for x, y in val_batches_it:
                x = x.to(device)
                y = y.to(device)
    
                out = model(x)[0]
                loss += criterion(out, y)
                accuracy += (torch.argmax(out, dim=1) == y).to(torch.float32).mean()
            loss /= len(val_batches_it)
            accuracy /= len(val_batches_it)
    
            epochs_it.set_postfix_str(f"cross_entropy_loss: {loss.item():.4f}, accuracy: {accuracy.item():.4f}")
                