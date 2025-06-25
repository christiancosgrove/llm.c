#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const float eps = 1e-5f;
    const float inv_C = 1.0f / C;
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            
            // Strategic prefetching based on tensor size
            if (C > 128) {
                _mm_prefetch((const char*)(x + 64), _MM_HINT_T0);
                _mm_prefetch((const char*)(x + 128), _MM_HINT_T0);
            }
            
            // Dual-pass approach with 16-element unrolling for better ILP
            __m256 sum_vec1 = _mm256_setzero_ps();
            __m256 sum_vec2 = _mm256_setzero_ps();
            __m256 sum_sq_vec1 = _mm256_setzero_ps();
            __m256 sum_sq_vec2 = _mm256_setzero_ps();
            
            int i = 0;
            // Process 16 elements at a time with dual AVX vectors
            for (; i <= C - 16; i += 16) {
                __m256 x_vec1 = _mm256_loadu_ps(&x[i]);
                __m256 x_vec2 = _mm256_loadu_ps(&x[i + 8]);
                
                sum_vec1 = _mm256_add_ps(sum_vec1, x_vec1);
                sum_vec2 = _mm256_add_ps(sum_vec2, x_vec2);
                sum_sq_vec1 = _mm256_fmadd_ps(x_vec1, x_vec1, sum_sq_vec1);
                sum_sq_vec2 = _mm256_fmadd_ps(x_vec2, x_vec2, sum_sq_vec2);
            }
            
            // Combine the dual vectors
            sum_vec1 = _mm256_add_ps(sum_vec1, sum_vec2);
            sum_sq_vec1 = _mm256_add_ps(sum_sq_vec1, sum_sq_vec2);
            
            // Process remaining 8 elements
            for (; i <= C - 8; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(&x[i]);
                sum_vec1 = _mm256_add_ps(sum_vec1, x_vec);
                sum_sq_vec1 = _mm256_fmadd_ps(x_vec, x_vec, sum_sq_vec1);
            }
            
            // Optimized horizontal reduction using more efficient operations
            // Use shuffle and add for better performance than hadd
            __m256 sum_perm = _mm256_permute2f128_ps(sum_vec1, sum_vec1, 1);
            sum_vec1 = _mm256_add_ps(sum_vec1, sum_perm);
            sum_vec1 = _mm256_hadd_ps(sum_vec1, sum_vec1);
            sum_vec1 = _mm256_hadd_ps(sum_vec1, sum_vec1);
            float sum = _mm256_cvtss_f32(sum_vec1);
            
            __m256 sum_sq_perm = _mm256_permute2f128_ps(sum_sq_vec1, sum_sq_vec1, 1);
            sum_sq_vec1 = _mm256_add_ps(sum_sq_vec1, sum_sq_perm);
            sum_sq_vec1 = _mm256_hadd_ps(sum_sq_vec1, sum_sq_vec1);
            sum_sq_vec1 = _mm256_hadd_ps(sum_sq_vec1, sum_sq_vec1);
            float sum_sq = _mm256_cvtss_f32(sum_sq_vec1);
            
            // Handle remaining elements
            for (; i < C; i++) {
                float xi = x[i];
                sum += xi;
                sum_sq += xi * xi;
            }
            
            float m = sum * inv_C;
            float v = sum_sq * inv_C - m * m;
            float s = 1.0f / sqrtf(v + eps);
            
            // Store mean and rstd early for better cache locality
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
            
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            
            // Optimized output calculation with AVX and prefetching
            __m256 m_vec = _mm256_set1_ps(m);
            __m256 s_vec = _mm256_set1_ps(s);
            
            // Simplified prefetch strategy for better cache efficiency
            if (C > 256) {
                _mm_prefetch((const char*)(weight + 128), _MM_HINT_T0);
                _mm_prefetch((const char*)(bias + 128), _MM_HINT_T0);
            }
            
            i = 0;
            for (; i <= C - 8; i += 8) {
                
                __m256 x_vec = _mm256_loadu_ps(&x[i]);
                __m256 weight_vec = _mm256_loadu_ps(&weight[i]);
                __m256 bias_vec = _mm256_loadu_ps(&bias[i]);
                
                // Compute normalized values: s * (x - m)
                __m256 norm_vec = _mm256_mul_ps(s_vec, _mm256_sub_ps(x_vec, m_vec));
                
                // Apply weight and bias: norm * weight + bias
                __m256 result = _mm256_fmadd_ps(norm_vec, weight_vec, bias_vec);
                
                _mm256_storeu_ps(&out_bt[i], result);
            }
            
            // Handle remaining elements with unrolling
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
            for (; i < C; i++) {
                float n = s * (x[i] - m);
                out_bt[i] = n * weight[i] + bias[i];
            }
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
