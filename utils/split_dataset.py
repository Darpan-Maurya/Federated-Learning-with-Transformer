# utils/split_dataset.py
import numpy as np
import os

def split_dataset(full_data_path, output_dir="data", num_clients=4):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(full_data_path)  # Shape: (N, window, d_model)
    total_samples = data.shape[0]
    split_size = total_samples // num_clients

    for i in range(num_clients):
        start = i * split_size
        end = (i + 1) * split_size if i < num_clients - 1 else total_samples
        np.save(os.path.join(output_dir, f"client_{i}.npy"), data[start:end])

    print(f"Split dataset into {num_clients} parts under `{output_dir}/`")
