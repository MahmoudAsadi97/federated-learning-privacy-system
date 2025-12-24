import torch


def mse(model, dataloader):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x)
            total_loss += loss_fn(preds, y).item()

    return total_loss / len(dataloader)
