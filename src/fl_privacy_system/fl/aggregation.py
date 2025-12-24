import torch


def fed_avg(state_dicts):
    """Federated averaging of model weights."""
    avg_state = {}

    for key in state_dicts[0].keys():
        avg_state[key] = torch.mean(
            torch.stack([sd[key] for sd in state_dicts]), dim=0
        )

    return avg_state
