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
            
            // Optimized 32-element unrolling with 4-way parallelism for maximum ILP
            __m256 sum_vec1 = _mm256_setzero_ps();
            __m256 sum_vec2 = _mm256_setzero_ps();
            __m256 sum_vec3 = _mm256_setzero_ps();
            __m256 sum_vec4 = _mm256_setzero_ps();
            __m256 sum_sq_vec1 = _mm256_setzero_ps();
            __m256 sum_sq_vec2 = _mm256_setzero_ps();
            __m256 sum_sq_vec3 = _mm256_setzero_ps();
            __m256 sum_sq_vec4 = _mm256_setzero_ps();
            
            int i = 0;
            // Process 32 elements at a time - removed prefetching for better cache efficiency
            for (; i <= C - 32; i += 32) {
                
                __m256 x_vec1 = _mm256_loadu_ps(&x[i]);
                __m256 x_vec2 = _mm256_loadu_ps(&x[i + 8]);
                __m256 x_vec3 = _mm256_loadu_ps(&x[i + 16]);
                __m256 x_vec4 = _mm256_loadu_ps(&x[i + 24]);
                
                sum_vec1 = _mm256_add_ps(sum_vec1, x_vec1);
                sum_vec2 = _mm256_add_ps(sum_vec2, x_vec2);
                sum_vec3 = _mm256_add_ps(sum_vec3, x_vec3);
                sum_vec4 = _mm256_add_ps(sum_vec4, x_vec4);
                
                sum_sq_vec1 = _mm256_fmadd_ps(x_vec1, x_vec1, sum_sq_vec1);
                sum_sq_vec2 = _mm256_fmadd_ps(x_vec2, x_vec2, sum_sq_vec2);
                sum_sq_vec3 = _mm256_fmadd_ps(x_vec3, x_vec3, sum_sq_vec3);
                sum_sq_vec4 = _mm256_fmadd_ps(x_vec4, x_vec4, sum_sq_vec4);
            }
            
            // Combine the 4 vectors efficiently
            sum_vec1 = _mm256_add_ps(_mm256_add_ps(sum_vec1, sum_vec2), _mm256_add_ps(sum_vec3, sum_vec4));
            sum_sq_vec1 = _mm256_add_ps(_mm256_add_ps(sum_sq_vec1, sum_sq_vec2), _mm256_add_ps(sum_sq_vec3, sum_sq_vec4));
            
            // Process remaining 8-element chunks
            for (; i <= C - 8; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(&x[i]);
                sum_vec1 = _mm256_add_ps(sum_vec1, x_vec);
                sum_sq_vec1 = _mm256_fmadd_ps(x_vec, x_vec, sum_sq_vec1);
            }
            
            // More efficient horizontal reduction using hadd intrinsics
            __m256 sum_hadd1 = _mm256_hadd_ps(sum_vec1, sum_vec1);
            __m256 sum_hadd2 = _mm256_hadd_ps(sum_hadd1, sum_hadd1);
            __m128 sum_lo = _mm256_extractf128_ps(sum_hadd2, 0);
            __m128 sum_hi = _mm256_extractf128_ps(sum_hadd2, 1);
            float sum = _mm_cvtss_f32(_mm_add_ss(sum_lo, sum_hi));
            
            __m256 sq_hadd1 = _mm256_hadd_ps(sum_sq_vec1, sum_sq_vec1);
            __m256 sq_hadd2 = _mm256_hadd_ps(sq_hadd1, sq_hadd1);
            __m128 sq_lo = _mm256_extractf128_ps(sq_hadd2, 0);
            __m128 sq_hi = _mm256_extractf128_ps(sq_hadd2, 1);
            float sum_sq = _mm_cvtss_f32(_mm_add_ss(sq_lo, sq_hi));
            
            // Handle remaining elements
            for (; i < C; i++) {
                float xi = x[i];
                sum += xi;
                sum_sq += xi * xi;
            }
            
            float m = sum * inv_C;
            float v = sum_sq * inv_C - m * m;
            float s = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(v + eps)));
            // Newton-Raphson refinement for better accuracy
            float half_v_eps = (v + eps) * 0.5f;
            s = s * (1.5f - half_v_eps * s * s);
            
            // Store mean and rstd early for better cache locality
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
            
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            
            // Optimized output calculation with 32-element unrolling
            __m256 m_vec = _mm256_set1_ps(m);
            __m256 s_vec = _mm256_set1_ps(s);
            
            i = 0;
            // Process 32 elements at a time for better throughput
            for (; i <= C - 32; i += 32) {
                __m256 x_vec1 = _mm256_loadu_ps(&x[i]);
                __m256 x_vec2 = _mm256_loadu_ps(&x[i + 8]);
                __m256 x_vec3 = _mm256_loadu_ps(&x[i + 16]);
                __m256 x_vec4 = _mm256_loadu_ps(&x[i + 24]);
                
                __m256 weight_vec1 = _mm256_loadu_ps(&weight[i]);
                __m256 weight_vec2 = _mm256_loadu_ps(&weight[i + 8]);
                __m256 weight_vec3 = _mm256_loadu_ps(&weight[i + 16]);
                __m256 weight_vec4 = _mm256_loadu_ps(&weight[i + 24]);
                
                __m256 bias_vec1 = _mm256_loadu_ps(&bias[i]);
                __m256 bias_vec2 = _mm256_loadu_ps(&bias[i + 8]);
                __m256 bias_vec3 = _mm256_loadu_ps(&bias[i + 16]);
                __m256 bias_vec4 = _mm256_loadu_ps(&bias[i + 24]);
                
                // Compute normalized values: s * (x - m)
                __m256 norm_vec1 = _mm256_mul_ps(s_vec, _mm256_sub_ps(x_vec1, m_vec));
                __m256 norm_vec2 = _mm256_mul_ps(s_vec, _mm256_sub_ps(x_vec2, m_vec));
                __m256 norm_vec3 = _mm256_mul_ps(s_vec, _mm256_sub_ps(x_vec3, m_vec));
                __m256 norm_vec4 = _mm256_mul_ps(s_vec, _mm256_sub_ps(x_vec4, m_vec));
                
                // Apply weight and bias: norm * weight + bias
                __m256 result1 = _mm256_fmadd_ps(norm_vec1, weight_vec1, bias_vec1);
                __m256 result2 = _mm256_fmadd_ps(norm_vec2, weight_vec2, bias_vec2);
                __m256 result3 = _mm256_fmadd_ps(norm_vec3, weight_vec3, bias_vec3);
                __m256 result4 = _mm256_fmadd_ps(norm_vec4, weight_vec4, bias_vec4);
                
                _mm256_storeu_ps(&out_bt[i], result1);
                _mm256_storeu_ps(&out_bt[i + 8], result2);
                _mm256_storeu_ps(&out_bt[i + 16], result3);
                _mm256_storeu_ps(&out_bt[i + 24], result4);
            }
            
            // Process remaining 8-element chunks
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
