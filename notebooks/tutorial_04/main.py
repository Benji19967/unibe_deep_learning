import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dataset import Digits
from IPython.display import clear_output
from mlp import MLP
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from tqdm.notebook import tqdm

HERE = Path(__file__)
DATA_DIR = (HERE / ".." / ".." / ".." / "data").resolve()
DATA_TUT4_DIR = DATA_DIR / "tutorial_04"
print(DATA_TUT4_DIR)

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")


def flatten(first, second):
    # first: (N, 28, 28) -> (N, 784)
    first_flat = first.view(first.size(0), -1)
    # second: (N, 28, 28) -> (N, 784)
    second_flat = second.view(second.size(0), -1)

    return torch.cat([first_flat, second_flat], dim=1)


def train_epoch(model: MLP, train_dataloader: DataLoader, optimizer, loss_fn):
    model.train()
    losses = []
    correct_predictions = 0
    # Iterate mini batches over training dataset
    for first, second, labels in tqdm(train_dataloader):
        first, second, labels = first.to(device), second.to(device), labels.to(device)
        optimizer.zero_grad()

        # concat images and pass to model
        output = model(flatten(first, second))

        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        # Log metrics
        losses.append(loss.item())
        predicted_labels = output.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
    accuracy = correct_predictions / len(train_dataloader.dataset)
    # Return loss values for each iteration and accuracy
    mean_loss = np.array(losses).mean()
    return mean_loss, accuracy


def evaluate(model, dataloader, loss_fn):
    """
    Compute loss on the validation or test set.
    """
    model.eval()
    loss = 1
    correct = 0
    losses = []
    with torch.no_grad():
        for first, second, labels in dataloader:
            first, second, labels = (
                first.to(device),
                second.to(device),
                labels.to(device),
            )
            logits = model(flatten(first, second))
            loss += loss_fn(logits, labels).item()
            pred = logits.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()

    loss /= len(dataloader)
    losses.append(loss)
    accuracy = correct / len(dataloader.dataset)
    print(
        "\nEvaluation: Avg. loss: {:.4f}, Accuracy: {} ({:.0f}%)\n".format(
            loss,
            accuracy,
            100.0 * correct / len(dataloader.dataset),
        ),
        flush=True,
    )

    mean_loss = np.array(losses).mean()
    return mean_loss, accuracy


def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_function):
    # We will monitor loss functions as the training progresses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.5,
    )

    for epoch in range(n_epochs):
        train_mean_loss, train_accuracy = train_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_function,
        )
        validation_mean_loss, val_accuracy = evaluate(
            model, val_dataloader, loss_function
        )
        train_losses.append(train_mean_loss)
        val_losses.append(validation_mean_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        scheduler.step()

        print(
            "Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}".format(
                epoch + 1,
                n_epochs,
                train_losses[-1],
                train_accuracies[-1],
                val_losses[-1],
                val_accuracies[-1],
            )
        )
    return train_losses, val_losses, train_accuracies, val_accuracies


def main():
    digits = Digits(root=str(DATA_TUT4_DIR), transforms=(ToTensor()))

    train_set, val_set, test_set = random_split(digits, lengths=[0.8, 0.1, 0.1])

    # make train, validation and test dataloaders
    batch_size = 64

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # train the model

    model = MLP().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-4
    )
    num_epochs = 25
    train_losses, val_losses, train_accuracies, val_accuracies = train(
        model, train_loader, val_loader, optimizer, num_epochs, nn.CrossEntropyLoss()
    )

    print("Evaluating test set")
    test_mean_loss, test_accuracy = evaluate(model, test_loader, nn.CrossEntropyLoss())


if __name__ == "__main__":
    main()
