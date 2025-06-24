
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
    int iterations = 1000;
    
    float* x = (float*) malloc(B * T * C * sizeof(float));
    float* w = (float*) malloc(C * sizeof(float));
    float* b = (float*) malloc(C * sizeof(float));
    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* mean = (float*) malloc(B * T * sizeof(float));
    float* rstd = (float*) malloc(B * T * sizeof(float));
    
    FILE *file = fopen("ln.bin", "rb");
    if (file == NULL) {
        printf("Error opening file\n");
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
    printf("%.6f\n", cpu_time_used);
    
    free(x); free(w); free(b); free(out); free(mean); free(rstd);
    return 0;
}
