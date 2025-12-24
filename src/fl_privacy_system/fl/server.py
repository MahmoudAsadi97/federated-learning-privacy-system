from copy import deepcopy
from .aggregation import fed_avg


class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_states):
        new_state = fed_avg(client_states)
        self.global_model.load_state_dict(new_state)

    def distribute(self):
        return deepcopy(self.global_model)
