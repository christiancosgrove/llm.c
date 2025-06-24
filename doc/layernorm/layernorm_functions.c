#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void layernorm_forward(float* restrict out, float* restrict mean, float* restrict rstd,
                       float* restrict inp, float* restrict weight, float* restrict bias,
                       int B, int T, int C) {
    const float eps = 1e-5f;
    const float inv_C = 1.0f / C;
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* restrict x = inp + b * T * C + t * C;
            float* restrict out_bt = out + b * T * C + t * C;
            
            // Single pass: calculate mean and variance together
            // Use multiple accumulators to reduce dependencies
            float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
            float sum_sq1 = 0.0f, sum_sq2 = 0.0f, sum_sq3 = 0.0f, sum_sq4 = 0.0f;
            
            // Unroll by 8 with 4-way accumulator to reduce dependencies
            int i = 0;
            for (; i < C - 7; i += 8) {
                float x0 = x[i], x1 = x[i+1], x2 = x[i+2], x3 = x[i+3];
                float x4 = x[i+4], x5 = x[i+5], x6 = x[i+6], x7 = x[i+7];
                
                sum1 += x0 + x4; sum2 += x1 + x5; sum3 += x2 + x6; sum4 += x3 + x7;
                sum_sq1 += x0*x0 + x4*x4; sum_sq2 += x1*x1 + x5*x5;
                sum_sq3 += x2*x2 + x6*x6; sum_sq4 += x3*x3 + x7*x7;
            }
            
            float sum = sum1 + sum2 + sum3 + sum4;
            float sum_sq = sum_sq1 + sum_sq2 + sum_sq3 + sum_sq4;
            
            // Handle remaining elements
            for (; i < C; i++) {
                float xi = x[i];
                sum += xi;
                sum_sq += xi * xi;
            }
            
            float m = sum * inv_C;
            float v = sum_sq * inv_C - m * m;
            float s = 1.0f / sqrtf(v + eps);
            
            // Apply normalization with unrolling
            i = 0;
            for (; i < C - 3; i += 4) {
                float n0 = s * (x[i] - m);
                float n1 = s * (x[i+1] - m);
                float n2 = s * (x[i+2] - m);
                float n3 = s * (x[i+3] - m);
                
                out_bt[i] = n0 * weight[i] + bias[i];
                out_bt[i+1] = n1 * weight[i+1] + bias[i+1];
                out_bt[i+2] = n2 * weight[i+2] + bias[i+2];
                out_bt[i+3] = n3 * weight[i+3] + bias[i+3];
            }
            // Handle remaining elements
            for (; i < C; i++) {
                float n = s * (x[i] - m);
                out_bt[i] = n * weight[i] + bias[i];
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
