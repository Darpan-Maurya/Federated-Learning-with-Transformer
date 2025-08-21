# run_simulation.py

import flwr as fl
import torch
import os
from client import client_fn
from task import get_model
from flwr.common import parameters_to_ndarrays

class ModelExportStrategy(fl.server.strategy.FedAvg):
    """Custom strategy that exports the final model"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None
        self.final_eval_loss = None
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and store final parameters"""
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            parameters, _ = aggregated
            # Convert Flower Parameters to list of numpy arrays
            self.final_parameters = parameters_to_ndarrays(parameters)
            print(f"Stored parameters from round {server_round}")
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Capture aggregated evaluation loss from the final round"""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is not None:
            try:
                # Handle different result structures
                if isinstance(aggregated, tuple) and len(aggregated) >= 2:
                    loss, _ = aggregated
                    if loss is not None:
                        self.final_eval_loss = float(loss)
                        print(f"Aggregated eval loss at round {server_round}: {self.final_eval_loss}")
                    else:
                        print(f"Warning: No loss value in evaluation results at round {server_round}")
                        self.final_eval_loss = 0.0
                else:
                    print(f"Warning: Unexpected evaluation result structure at round {server_round}: {type(aggregated)}")
                    self.final_eval_loss = 0.0
            except Exception as e:
                print(f"Warning: Error processing evaluation results at round {server_round}: {e}")
                self.final_eval_loss = 0.0
        else:
            print(f"Warning: No evaluation results at round {server_round}")
            self.final_eval_loss = 0.0
        
        return aggregated
    
    def export_final_model(self):
        """Export the final model with the stored parameters"""
        if self.final_parameters is None:
            print("No final parameters available for export")
            return
        
        try:
            # Create output directory
            os.makedirs("exported_models", exist_ok=True)
            
            # Get model structure from first client
            from task import partition_data
            train_data, val_data = partition_data(0, 2)
            input_dim = train_data[0][0].shape[1]
            model = get_model(input_dim)
            
            # Convert numpy parameters back to torch tensors
            state_dict = model.state_dict()
            new_state_dict = {}
            for (key, _), param in zip(state_dict.items(), self.final_parameters):
                new_state_dict[key] = torch.tensor(param)
            
            # Load the final parameters into the model
            model.load_state_dict(new_state_dict)
            
            # Safely handle final_eval_loss
            eval_loss = self.final_eval_loss if self.final_eval_loss is not None else 0.0
            
            # Save the complete model
            model_path = "exported_models/final_transformer_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'd_model': input_dim,
                    'N_layers': 2,
                    'attention': 'single',
                    'window': 50,
                    'dropout': 0,
                    'd_ff': 64
                },
                'training_info': {
                    'num_rounds': 3,
                    'num_clients': 2,
                    'parameters_shape': [p.shape for p in self.final_parameters],
                    'final_eval_loss': eval_loss
                }
            }, model_path)
            
            print(f"Final model exported successfully to: {model_path}")
            
        except Exception as e:
            print(f"Error exporting model: {e}")

def main():
    # Create custom strategy
    strategy = ModelExportStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    
    # Start the simulation with reduced resources
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Export the final model
    strategy.export_final_model()

if __name__ == "__main__":
    main()
