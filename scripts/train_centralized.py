import pandas as pd
import torch
from torch.utils.data import DataLoader
from fl_privacy_system.models.model import EnergyRegressor
from fl_privacy_system.utils.dataset import make_supervised
from fl_privacy_system.utils.seed import set_global_seed



def main():
    set_global_seed(42)

    df = pd.read_csv("data/raw/energy.csv", index_col=0, parse_dates=True)
    dataset = make_supervised(df["target"], window=1)

    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = EnergyRegressor(input_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(5):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: MSE={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "centralized_model.pt")
    print("Saved centralized baseline model")


if __name__ == "__main__":
    main()
