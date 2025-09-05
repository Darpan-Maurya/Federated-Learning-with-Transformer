import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from .layers import Norm, EncoderLayer, DecoderLayer, get_clones
from .attention import PositionalEncoder
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout, d_ff):
        super().__init__()
        self.N_layers = N_layers
        self.pe = PositionalEncoder(d_model, window, device)
        self.layers = get_clones(EncoderLayer(d_model, attention, device, dropout, d_ff), N_layers)
        self.norm = Norm(d_model, device)
    def forward(self, src):
        # En la variable x se almacena src, el batch de datos de entrada en cada iteración,
        # pero con el positional encoding aplicado mediante una suma
        x = self.pe(src)
        # Seguidamente se pasa x por las N capas del encoder (que son todas idénticas)
        for i in range(self.N_layers):
            x = self.layers[i](x)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout, d_ff):
        super().__init__()
        self.N_layers = N_layers
        self.pe = PositionalEncoder(d_model, window, device)
        self.layers = get_clones(DecoderLayer(d_model, attention, device, dropout, d_ff), N_layers)
        self.norm = Norm(d_model, device)
    def forward(self, trg, e_outputs, mask):
        # x = self.embed(trg)
        x = self.pe(trg)
        for i in range(self.N_layers):
            x = self.layers[i](x, e_outputs, mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, d_model, N_layers, attention, window, device, dropout=0.1, d_ff = 512):
        super().__init__()
        self.encoder = Encoder(d_model, N_layers, attention, window, device, dropout, d_ff)
        self.decoder = Decoder(d_model, N_layers, attention, window, device, dropout, d_ff)
        
        # Fix the linear layer dimensions for network flow data
        # Input: d_model * window (43 * 50 = 2150)
        # Output: 1 (binary classification: normal vs attack)
        self.out = nn.Linear(d_model * window, 1).to(torch.device(device))
        
        self.window = window
        self.device = device
        
    def nopeak_mask(self, size):
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        np_mask = Variable(torch.from_numpy(np_mask) == 0).to(torch.device(self.device))
        return np_mask
        
    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, trg):
        # Debug dimensions
        print(f"Forward - src shape: {src.shape}, trg shape: {trg.shape}")
        
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, self.nopeak_mask(trg.size(1)))
        
        # Reshape decoder output for linear layer
        batch_size = d_output.size(0)
        d_output_flat = d_output.view(batch_size, -1)
        print(f"Decoder output flat shape: {d_output_flat.shape}")
        
        output = self.out(d_output_flat)
        print(f"Final output shape: {output.shape}")
        
        return output
    
    def train_model(self, data_loader, epochs=10, print_every=1, return_evo=False):
        optim = torch.optim.Adam(self.parameters(), lr=0.00001, betas=(0.9, 0.98))
        self.train()    
        total_loss = 0
        loss_evo = []
        
        for epoch in range(epochs):
            print(f'\nEpoch: {epoch+1} of {epochs}')
            with tqdm(total = data_loader.__len__()) as pbar:
                for i, batch in enumerate(data_loader):   
                    # Debug batch structure
                    if i == 0:  # Only print first batch info
                        print(f"Batch type: {type(batch)}")
                        print(f"Batch length: {len(batch)}")
                        if len(batch) == 2:
                            print(f"Features shape: {batch[0].shape}")
                            print(f"Targets shape: {batch[1].shape}")
                        else:
                            print(f"Single tensor shape: {batch[0].shape}")
                    
                    # Handle both feature+target and single tensor cases
                    if len(batch) == 2:
                        features, targets = batch
                    else:
                        features = batch[0]
                        # For single tensor, use last timestep as target
                        targets = features[:, -1, -1]  # Last feature of last timestep
                    
                    pbar.update(1)
                    
                    # Check if we have enough dimensions
                    if features.dim() < 3:
                        print(f"Error: features dimension too low: {features.shape}")
                        continue
                        
                    # For network flow data: use full sequence as input
                    # Target is binary attack label
                    preds = self.forward(features, features)  # Use same sequence for src and trg
                    
                    # Ensure targets are the right shape for binary classification
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(1)  # Add batch dimension
                    
                    # Use Binary Cross Entropy for attack detection
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(preds, targets.float())           
                    loss.backward()
                    optim.step()
                    
                    total_loss += loss.item()
                    loss_evo.append(loss.item())

                    if (i + 1) % print_every == 0:
                        loss_avg = total_loss / print_every
                        pbar.set_postfix({'Last mean loss': loss_avg})
                        total_loss = 0
                        
        if return_evo:
            return loss_evo

    def detect(self, data_loader):
        self.eval()    
        ano_scores = []
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        with tqdm(total = data_loader.__len__()) as pbar:
            for batch in data_loader:  
                # Handle both feature+target and single tensor cases
                if len(batch) == 2:
                    features, targets = batch
                else:
                    features = batch[0]
                    targets = features[:, -1, -1]  # Last feature of last timestep
                
                pbar.update(1)
                
                # Get predictions
                preds = self.forward(features, features)
                
                # Calculate anomaly scores (higher = more likely attack)
                # Use the raw logits as anomaly scores
                scores = torch.sigmoid(preds).squeeze()
                ano_scores.extend(scores.cpu().tolist())
                
        return ano_scores