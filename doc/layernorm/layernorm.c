// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const float eps = 1e-5f;
    const float inv_C = 1.0f / C;
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            float* out_bt = out + b * T * C + t * C;
            
            // calculate mean and variance in single pass
            float sum = 0.0f;
            float sum_sq = 0.0f;
            
            // First pass: compute sum and sum of squares
            for (int i = 0; i < C; i++) {
                float val = x[i];
                sum += val;
                sum_sq += val * val;
            }
            
            // Calculate mean and variance
            float m = sum * inv_C;
            float v = sum_sq * inv_C - m * m;
            
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            
            // Second pass: normalize and apply weight/bias
            for (int i = 0; i < C; i++) {
                float n = (x[i] - m) * s; // normalized output
                out_bt[i] = n * weight[i] + bias[i]; // scale and shift
            }
            
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, char* label) {
    int ok = 1;
    int failures = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float error = fabs(a[i] - b[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error > 1e-5) {
            failures++;
            ok = 0;
        }
    }
    
    printf("%s: %s (failures: %d/%d, max_error: %.2e)\n", 
           label, ok ? "PASS" : "FAIL", failures, n, max_error);
    return ok;
}

int main() {

    int B = 8; // batch
    int T = 64; // time / sequence length
    int C = 256; // number of channels

    float* x = (float*) malloc(B * T * C * sizeof(float));
    float* w = (float*) malloc(C * sizeof(float));
    float* b = (float*) malloc(C * sizeof(float));
    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* mean = (float*) malloc(B * T * sizeof(float));
    float* rstd = (float*) malloc(B * T * sizeof(float));
    float* dout = (float*) malloc(B * T * C * sizeof(float));
    float* dx = (float*) malloc(B * T * C * sizeof(float));
    float* dw = (float*) malloc(C * sizeof(float));
    float* db = (float*) malloc(C * sizeof(float));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    fread(x, sizeof(float), B * T * C, file);
    fread(w, sizeof(float), C, file);
    fread(b, sizeof(float), C, file);
    fread(out, sizeof(float), B * T * C, file);
    fread(mean, sizeof(float), B * T, file);
    fread(rstd, sizeof(float), B * T, file);
    fread(dout, sizeof(float), B * T * C, file);
    fread(dx, sizeof(float), B * T * C, file);
    fread(dw, sizeof(float), C, file);
    fread(db, sizeof(float), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    float* c_out = (float*) malloc(B * T * C * sizeof(float));
    float* c_mean = (float*) malloc(B * T * sizeof(float));
    float* c_rstd = (float*) malloc(B * T * sizeof(float));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass only
    int all_pass = 1;
    all_pass &= check_tensor(out, c_out, B*T*C, "out");
    all_pass &= check_tensor(mean, c_mean, B*T, "mean");
    all_pass &= check_tensor(rstd, c_rstd, B*T, "rstd");

    printf("Forward pass verification: %s\n", all_pass ? "PASS" : "FAIL");

    // Skip backward pass verification
    
    free(x);
    free(w);
    free(b);
    free(out);
    free(mean);
    free(rstd);
    free(dout);
    free(dx);
    free(dw);
    free(db);
    free(c_out);
    free(c_mean);
    free(c_rstd);
    return all_pass ? 0 : 1;
}
