#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <emmintrin.h>  // SSE2 intrinsics

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const float eps = 1e-5f;
    const float inv_C = 1.0f / C;
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            
            // SIMD-optimized statistics calculation
            __m128 sum_vec = _mm_setzero_ps();
            __m128 sum_sq_vec = _mm_setzero_ps();
            
            int i = 0;
            // Process 4 elements at a time with SIMD
            for (; i <= C - 4; i += 4) {
                __m128 x_vec = _mm_loadu_ps(&x[i]);
                sum_vec = _mm_add_ps(sum_vec, x_vec);
                sum_sq_vec = _mm_add_ps(sum_sq_vec, _mm_mul_ps(x_vec, x_vec));
            }
            
            // Horizontal sum for SIMD results
            float sum_array[4], sum_sq_array[4];
            _mm_storeu_ps(sum_array, sum_vec);
            _mm_storeu_ps(sum_sq_array, sum_sq_vec);
            
            float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
            float sum_sq = sum_sq_array[0] + sum_sq_array[1] + sum_sq_array[2] + sum_sq_array[3];
            
            // Handle remaining elements
            for (; i < C; i++) {
                float xi = x[i];
                sum += xi;
                sum_sq += xi * xi;
            }
            
            float m = sum * inv_C;
            float v = sum_sq * inv_C - m * m;
            
            // Fast inverse square root approximation for better performance
            float rsqrt_v_eps = 1.0f / sqrtf(v + eps);
            
            // Store mean and rstd early for better cache locality
            mean[b * T + t] = m;
            rstd[b * T + t] = rsqrt_v_eps;
            
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            
            // SIMD-optimized output calculation
            __m128 m_vec = _mm_set1_ps(m);
            __m128 s_vec = _mm_set1_ps(rsqrt_v_eps);
            
            i = 0;
            for (; i <= C - 4; i += 4) {
                __m128 x_vec = _mm_loadu_ps(&x[i]);
                __m128 weight_vec = _mm_loadu_ps(&weight[i]);
                __m128 bias_vec = _mm_loadu_ps(&bias[i]);
                
                // Normalize: (x - m) * s
                __m128 norm_vec = _mm_mul_ps(_mm_sub_ps(x_vec, m_vec), s_vec);
                
                // Apply weight and bias: norm * weight + bias
                __m128 result_vec = _mm_add_ps(_mm_mul_ps(norm_vec, weight_vec), bias_vec);
                
                _mm_storeu_ps(&out_bt[i], result_vec);
            }
            
            // Handle remaining elements
            for (; i < C; i++) {
                float n = rsqrt_v_eps * (x[i] - m);
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
