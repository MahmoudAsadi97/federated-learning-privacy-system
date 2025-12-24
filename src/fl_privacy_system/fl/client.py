import torch
from torch.utils.data import DataLoader
from copy import deepcopy

from opacus import PrivacyEngine


class FLClient:
    def __init__(
        self,
        client_id,
        model,
        data,
        lr=1e-3,
        epochs=1,
        batch_size=128,
        dp=False,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
    ):
        self.client_id = client_id
        self.model = deepcopy(model)
        self.data = data
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dp = dp
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.epsilon = None

    def train(self):
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        loader = DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        privacy_engine = None

        if self.dp:
            privacy_engine = PrivacyEngine()
            self.model, optimizer, loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )

        for _ in range(self.epochs):
            for x, y in loader:
                optimizer.zero_grad()
                preds = self.model(x)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()

        if self.dp:
            self.epsilon = privacy_engine.get_epsilon(delta=1e-5)

            # IMPORTANT: unwrap GradSampleModule before sending to server
            clean_state = {
                k.replace("_module.", ""): v.detach().cpu()
                for k, v in self.model.state_dict().items()
            }
        else:
            clean_state = {
                k: v.detach().cpu()
                for k, v in self.model.state_dict().items()
            }

        return clean_state
