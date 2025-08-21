# Federated Learning with Flower and Transformers

This project demonstrates a simple **Federated Learning (FL)** setup using [Flower (flwr)](https://flower.dev/) with a lightweight **Transformer model** built in PyTorch.

---

## 🚀 Project Structure

- **client.py** → Defines the Flower client logic (training, evaluation, parameter exchange).  
- **server.py** → Runs the Flower server with a FedAvg strategy.  
- **task.py** → Contains the Transformer model, training/evaluation routines, and data utilities.  
- **exported_models/** → Directory where trained models are saved.  

---

## 🧠 About the Transformer Model

This project uses a **custom Transformer model** tailored for time-series or tabular data in a federated learning setting.

### 🔹 Model Configuration
- **Input dimension (`d_model`)** → Based on dataset feature size  
- **Number of layers (`N`)** → 2 Transformer encoder layers  
- **Attention type** → Simplified *single-head self-attention*  
- **Window size** → 50 (sequence length considered)  
- **Dropout** → 0 (can be increased for regularization)  
- **Feedforward dimension (`d_ff`)** → 64  

### 🔹 Why Transformers?
- Capture **long-range dependencies** better than RNNs or LSTMs.  
- Flexible architecture that works well for sequential, tabular, and time-series data.  
- Scales naturally in a **federated setting**, since each client can adapt to its own data distribution.  

### 🔹 Model Export
At the end of federated training, the server saves the final global model in `exported_models/final_transformer_model.pth`.  
It includes:  
- Trained **model weights (`state_dict`)**  
- **Config** (model hyperparameters for reloading)  
- **Training info** (rounds, number of clients, final loss, etc.)  

You can reload the model later using PyTorch’s `torch.load`.

---

## ▶️ Running the Project

1. Start the server:
   ```bash
   python server.py
   ```

2. In separate terminals, start clients (e.g., 2 clients):
   ```bash
   python client.py
   python client.py
   ```

The server will coordinate training for **3 rounds** (as configured in `server.py`).

---
## ⚙️ Installation

1. **Clone the repo**:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   flwr==1.7.0
   torch
   numpy
   ```

---

## 🚀 Usage

### 1. Run with a Real Federated Server
Start the server:
```bash
python server.py
```

Start each client (in separate terminals):
```bash
python client.py
```

- The server coordinates rounds of training.
- Clients perform local training and send updates back.

---

### 2. Run with Local Simulation
Instead of spinning up real clients, you can simulate everything locally:
```bash
python run_simulation.py
```

This:
- Spawns multiple virtual clients.
- Runs federated training for a fixed number of rounds.
- Exports the final trained model.

---
## 💾 Model Export

After simulation, the trained model is saved to:

```
exported_models/final_transformer_model.pth
```

The saved file includes:
- Model weights (`state_dict`)
- Model configuration (`d_model`, `N_layers`, etc.)
- Training info (rounds, clients, evaluation loss)

You can later load it in PyTorch:
```python
import torch

checkpoint = torch.load("exported_models/final_transformer_model.pth")
model_state = checkpoint['model_state_dict']
model_config = checkpoint['model_config']
```

---

## ⚡ Features

- Federated Averaging (`FedAvg`) strategy.
- Support for **real clients** (via `server.py`) and **local simulation** (`run_simulation.py`).
- Custom strategy with **automatic final model export**.
- Lightweight and easy to extend with custom datasets/models.

---

## 📂 Outputs

- Trained model is saved to: `exported_models/final_transformer_model.pth`  
- Training/evaluation logs printed per client  
- Metrics such as **MSE loss** are reported back to the server  

---


## ✅ Summary

This project combines **Federated Learning + Transformers** using Flower, enabling decentralized model training while preserving data privacy.
