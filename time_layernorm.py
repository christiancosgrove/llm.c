import torch
import time
import json

# Import the LayerNorm class from the original file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'doc', 'layernorm'))
from layernorm import LayerNorm

def profile_layernorm():
    # Set up test data similar to the original
    B = 2
    T = 3  
    C = 4
    
    # Create input tensors
    x = torch.randn(B, T, C)
    w = torch.randn(C)
    b = torch.randn(C)
    
    # Warm up run to avoid cold start effects
    for _ in range(10):
        out, cache = LayerNorm.forward(x, w, b)
    
    # Time 1000 forward passes
    start_time = time.time()
    for _ in range(1000):
        out, cache = LayerNorm.forward(x, w, b)
    end_time = time.time()
    
    total_time = end_time - start_time
    return total_time

if __name__ == "__main__":
    # Run the profiling
    execution_time = profile_layernorm()
    
    # Create the results dictionary
    results = {"score": execution_time}
    
    # Write results to optimization_results.json
    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Profiling complete. Time for 1000 forward passes: {execution_time:.6f} seconds")
    print(f"Results written to optimization_results.json")
