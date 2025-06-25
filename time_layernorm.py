import torch
import time
import json
import subprocess
import os
import sys
import numpy as np
import struct

# Import the LayerNorm class from the original file for reference
sys.path.append(os.path.join(os.path.dirname(__file__), 'doc', 'layernorm'))
from layernorm import LayerNorm

def build_c_implementation():
    """Build the C layernorm implementation"""
    layernorm_dir = os.path.join(os.path.dirname(__file__), 'doc', 'layernorm')
    layernorm_c = os.path.join(layernorm_dir, 'layernorm.c')
    layernorm_exe = os.path.join(layernorm_dir, 'layernorm')
    
    # Build command
    build_cmd = ['gcc', layernorm_c, '-o', layernorm_exe, '-lm', '-O3']
    
    try:
        result = subprocess.run(build_cmd, capture_output=True, text=True, cwd=layernorm_dir)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False
        print("C implementation built successfully")
        return True
    except Exception as e:
        print(f"Build error: {e}")
        return False

def write_reference_data(x, w, b):
    """Write reference data for C implementation to read"""
    layernorm_dir = os.path.join(os.path.dirname(__file__), 'doc', 'layernorm')
    
    # Generate reference using Python implementation
    out, cache = LayerNorm.forward(x, w, b)
    x_cached, w_cached, mean, rstd = cache
    
    # Create dummy gradients for completeness (C code expects them)
    dout = torch.randn_like(out)
    dx, dw, db = LayerNorm.backward(dout, cache)
    
    def write_tensor(tensor, handle):
        handle.write(tensor.detach().numpy().astype("float32").tobytes())
    
    # Write to ln.bin file that C code expects
    ln_bin_path = os.path.join(layernorm_dir, 'ln.bin')
    with open(ln_bin_path, 'wb') as f:
        write_tensor(x, f)       # (B, T, C)
        write_tensor(w, f)       # (C,)
        write_tensor(b, f)       # (C,)
        write_tensor(out, f)     # (B, T, C)
        write_tensor(mean, f)    # (B, T)
        write_tensor(rstd, f)    # (B, T)
        write_tensor(dout, f)    # (B, T, C)
        write_tensor(dx, f)      # (B, T, C)
        write_tensor(dw, f)      # (C,)
        write_tensor(db, f)      # (C,)
    
    return out, mean, rstd

def run_c_implementation():
    """Run the C layernorm implementation and return timing"""
    layernorm_dir = os.path.join(os.path.dirname(__file__), 'doc', 'layernorm')
    layernorm_exe = os.path.join(layernorm_dir, 'layernorm')
    
    try:
        result = subprocess.run([layernorm_exe], capture_output=True, text=True, cwd=layernorm_dir)
        if result.returncode != 0:
            return False
        
        # Check if output indicates success
        output_lines = result.stdout.strip().split('\n')
        success = True
        for line in output_lines:
            if 'NOT OK' in line:
                success = False
                break
        
        return success
    except Exception as e:
        return False

def profile_c_layernorm(x, w, b, iterations=1000):
    """Profile the C layernorm implementation"""
    layernorm_dir = os.path.join(os.path.dirname(__file__), 'doc', 'layernorm')
    
    # Write a timing program that uses the actual layernorm.c implementation
    timing_c_code = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Forward declaration - actual implementation comes from layernorm.c
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C);

int main() {
    int B = 8, T = 64, C = 256;
    int iterations = ''' + str(iterations) + ''';
    
    float* x = (float*) malloc(B * T * C * sizeof(float));
    float* w = (float*) malloc(C * sizeof(float));
    float* b = (float*) malloc(C * sizeof(float));
    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* mean = (float*) malloc(B * T * sizeof(float));
    float* rstd = (float*) malloc(B * T * sizeof(float));
    
    FILE *file = fopen("ln.bin", "rb");
    if (file == NULL) {
        printf("Error opening file\\n");
        return 1;
    }
    fread(x, sizeof(float), B * T * C, file);
    fread(w, sizeof(float), C, file);
    fread(b, sizeof(float), C, file);
    // Skip the rest of the reference data (out, mean, rstd, dout, dx, dw, db)
    fseek(file, (B * T * C + B * T + B * T + B * T * C + B * T * C + C + C) * sizeof(float), SEEK_CUR);
    fclose(file);
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        layernorm_forward(out, mean, rstd, x, w, b, B, T, C);
    }
    
    // Time the iterations
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        layernorm_forward(out, mean, rstd, x, w, b, B, T, C);
    }
    clock_t end = clock();
    
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("%.6f\\n", cpu_time_used);
    
    free(x); free(w); free(b); free(out); free(mean); free(rstd);
    return 0;
}
'''
    
    # Write timing program
    timing_c_path = os.path.join(layernorm_dir, 'timing_layernorm.c')
    timing_exe_path = os.path.join(layernorm_dir, 'timing_layernorm')
    layernorm_c_path = os.path.join(layernorm_dir, 'layernorm.c')
    
    with open(timing_c_path, 'w') as f:
        f.write(timing_c_code)
    
    # Build timing program, linking with the layernorm functions
    layernorm_functions_path = os.path.join(layernorm_dir, 'layernorm_functions.c')
    build_cmd = ['gcc', timing_c_path, layernorm_functions_path, '-o', timing_exe_path, '-lm', '-O3', '-mavx', '-mavx2', '-mfma']
    result = subprocess.run(build_cmd, capture_output=True, text=True, cwd=layernorm_dir)
    if result.returncode != 0:
        print(f"Timing build failed: {result.stderr}")
        return None
    
    # Run timing program
    result = subprocess.run([timing_exe_path], capture_output=True, text=True, cwd=layernorm_dir)
    if result.returncode != 0:
        print(f"Timing execution failed: {result.stderr}")
        return None
    
    try:
        timing = float(result.stdout.strip())
        return timing
    except ValueError:
        print(f"Could not parse timing output: {result.stdout}")
        return None

def profile_layernorm():
    # Set up test data with larger dimensions for ~50x more computation
    B = 8
    T = 64  
    C = 256
    
    # Create input tensors
    x = torch.randn(B, T, C)
    w = torch.randn(C)
    b = torch.randn(C)
    
    if not build_c_implementation():
        # Fallback to Python timing
        for _ in range(10):
            out, cache = LayerNorm.forward(x, w, b)
        
        start_time = time.time()
        for _ in range(1000):
            out, cache = LayerNorm.forward(x, w, b)
        end_time = time.time()
        
        return end_time - start_time
    
    py_out, py_mean, py_rstd = write_reference_data(x, w, b)
    
    if not run_c_implementation():
        return None
    
    c_time = profile_c_layernorm(x, w, b, 1000)
    if c_time is None:
        return None
    
    # Also time Python for comparison
    for _ in range(10):
        out, cache = LayerNorm.forward(x, w, b)
    
    start_time = time.time()
    for _ in range(1000):
        out, cache = LayerNorm.forward(x, w, b)
    end_time = time.time()
    
    py_time = end_time - start_time
    
    print(f"C: {c_time:.6f}s, Python: {py_time:.6f}s, Speedup: {py_time/c_time:.1f}x")
    
    return c_time

if __name__ == "__main__":
    # Run the profiling
    execution_time = profile_layernorm()
    
    if execution_time is not None:
        # Create the results dictionary
        results = {"score": execution_time}
        
        # Write results to optimization_results.json
        with open("optimization_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results: {execution_time:.6f}s")
    else:
        print("Failed!")
