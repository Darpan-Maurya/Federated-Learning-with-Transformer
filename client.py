# client.py

import flwr as fl
import torch
from torch.utils.data import DataLoader
from task import get_model, load_data, train, test, partition_data
from flwr.common import Context

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 50

class FLClient(fl.client.NumPyClient):
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        
        # Check if data has targets (2 elements) or just features (1 element)
        if len(self.train_data[0]) == 2:
            # Data has features and targets
            input_dim = self.train_data[0][0].shape[1]
            print(f"Client initialized with features+targets, input_dim: {input_dim}")
        else:
            # Data has only features
            input_dim = self.train_data[0][0].shape[1]
            print(f"Client initialized with features only, input_dim: {input_dim}")
        
        self.model = get_model(input_dim)

    def get_parameters(self, config):
        # Ensure order consistency by using items()
        return [val.cpu().numpy() for key, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {}
        for (key, _), val in zip(state_dict.items(), parameters):
            new_state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(new_state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        train(self.model, train_loader, epochs=1)
        return self.get_parameters({}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loader = DataLoader(self.val_data, batch_size=32)
        loss = test(self.model, val_loader)
        return float(loss), len(val_loader.dataset), {"mse": float(loss)}


def client_fn(context: Context) -> fl.client.Client:
    # Extract client ID from context
    client_id = int(context.cid) if hasattr(context, 'cid') else 0
    num_clients = 2  # Make sure this matches your simulation setup
    train_data, val_data = partition_data(client_id, num_clients)
    return FLClient(train_data, val_data).to_client()

app = fl.client.ClientApp(client_fn)
