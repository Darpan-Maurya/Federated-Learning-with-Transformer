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

## 📂 Outputs

- Trained model is saved to: `exported_models/final_transformer_model.pth`  
- Training/evaluation logs printed per client  
- Metrics such as **MSE loss** are reported back to the server  

---

## 📌 Requirements

Install dependencies with:

```bash
pip install flwr torch
```

---

## ✅ Summary

This project combines **Federated Learning + Transformers** using Flower, enabling decentralized model training while preserving data privacy.
