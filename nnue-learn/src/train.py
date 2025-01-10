import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.nnue import NNUE


def load_checkpoint(
        model,
        optimizer,
        scheduler,
        filename="checkpoint.pth"):
    if not os.path.exists(filename):
        return 0
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

def save_layer_weights(weights: nn.Linear, filename):
    weight_matrix = weights.weight.cpu().data.numpy()  # shape [out_features, in_features]
    flat_weights = weight_matrix.flatten()  # shape [out_features * in_features]

    with open(filename, 'w') as file:
        file.write(','.join(str(x) for x in flat_weights))
        file.write('\n')


def save_nnue_weights(net: NNUE):
    save_layer_weights(net.fc1, "fc1.weights.csv")
    save_layer_weights(net.fc2, "fc2.weights.csv")

def validate(model: nn.Module, validation_dataloader: DataLoader):
    batches_length = 0
    criterion = nn.MSELoss()
    running_loss = 0.0

    for batch_idx, (batch_inputs, batch_scores) in enumerate(validation_dataloader):
        batches_length += 1
        batch_inputs = batch_inputs.to("mps")
        batch_scores = batch_scores.to("mps")
        outputs = model(batch_inputs)
        loss = criterion(outputs.squeeze(), batch_scores)
        running_loss += loss.item()

    return running_loss / batches_length

def save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        filename="checkpoint.pth"):
    checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    model.save_weights(epoch)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(
        model,
        optimizer,
        scheduler,
        filename="checkpoint.pth"):
    if not os.path.exists(filename):
        return 0
    checkpoint = torch.load(filename, weights_only=True)
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.load_weights(epoch)

    return epoch


def train(
        model: NNUE,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader
):
    device = torch.device("mps")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    epoch = load_checkpoint(model, optimizer, scheduler) + 1


    TRAIN_FILE = 'train.csv'
    with open(TRAIN_FILE, 'a') as train:
        train.write('Epoch,Train loss,Validate loss\n')

    while True:
        model.train()
        running_loss = 0.0
        count = 0
        index = 0

        for batch_idx, (batch_inputs, batch_scores) in enumerate(train_data_loader):
            index += 1
            if index % 100 == 0:
                print(f"Learning: {index}", flush=True)
            count += len(batch_inputs)
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_scores = batch_scores.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.squeeze(), batch_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        loss = (running_loss / index)
        scheduler.step(loss)

        validate_loss = validate(model, validation_data_loader)
        save_checkpoint(model, optimizer, scheduler, epoch)
        print(f"Epoch [{epoch}], Train loss: {loss:.4f}, Validate loss: {validate_loss:.4f}", flush=True)

        with open(TRAIN_FILE, 'a') as train:
            train.write(f"{epoch},{loss:.4f},{validate_loss:.4f}\n")

        epoch += 1

        if loss < 0.05:
            break
        print(optimizer.param_groups[0]['lr'])