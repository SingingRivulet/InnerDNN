#ifndef INNER_DNN_SHADERS
#define INNER_DNN_SHADERS

#include "innerDNN/gpu.h"

typedef struct {
    GLuint shader_rmsnorm_squares_and_sum;
    GLuint shader_sum;
    GLuint shader_sum_vec4;
    GLuint shader_rmsnorm_normalize_and_scale;
    GLuint shader_rmsnorm_normalize_and_scale_inplace;
    GLuint shader_accum;
    GLuint shader_positionalEncoding;
    GLuint shader_max;
    GLuint shader_max_vec4;
    GLuint shader_softmax_exp;
    GLuint shader_softmax_normalize;
    GLuint shader_transformer_silu_and_mulW3;
    GLuint shader_transformer_get_query_vector;
    GLuint shader_transformer_build_attMat;
    GLuint shader_transformer_softmax_input;
    GLuint shader_transformer_softmax_output;
    GLuint shader_temperature;
    GLuint shader_copyBuffer;
    GLuint shader_matmul;
    GLuint shader_matmul_trans_vec4;
    GLuint shader_sigmoid;
    GLuint shader_reluAndsqr;
    GLuint shader_variance_before_sum;
    GLuint shader_rwkv_att_rkv;
    GLuint shader_rwkv_att_wkv;
    GLuint shader_rwkv_ffn;
    GLuint shader_layerNorm_inplace;
    GLuint shader_layerNorm;
} shaderPrograms;

void innerDNN_shaders_createProgram(shaderPrograms* program);
void innerDNN_shaders_deleteProgram(shaderPrograms* prog);

//归约法
void innerDNN_shaders_reduce_step(GLuint kernel,
                                  GLuint inBuffer,
                                  int insize,
                                  GLuint outBuffer,
                                  int outsize,
                                  int numSeq);
GLuint innerDNN_shaders_reduce_iteration(GLuint kernel_step,
                                         GLuint kernel_step_v4,
                                         GLuint data,
                                         GLuint cache_1,
                                         int insize,
                                         int numSeq,
                                         GLuint* otherBuffer,
                                         GLuint* outputAt);
GLuint innerDNN_shaders_reduce_iteration_input(GLuint kernel_step,
                                               GLuint kernel_step_v4,
                                               GLuint kernel_step_input,
                                               GLuint data,
                                               GLuint cache_1,
                                               GLuint cache_2,
                                               int insize,
                                               int numSeq,
                                               GLuint* otherBuffer,
                                               GLuint* outputAt);

//矩阵与向量之间的乘法
void innerDNN_shaders_matxvec(
    shaderPrograms* prog,
    GLuint xout,
    GLuint x,
    GLuint w,
    int n,
    int d,
    int x_offset,
    int w_offset);
void innerDNN_shaders_matxvec_trans_vec4(
    shaderPrograms* prog,
    GLuint xout,
    GLuint x,
    GLuint w,
    int n,
    int d,
    int x_offset,
    int w_offset);

void innerDNN_shaders_copyBuffer(shaderPrograms* prog, GLuint src, GLuint dst, int src_offset, int dst_offset, int size);

//一些基本操作
void innerDNN_shaders_accum(shaderPrograms* prog, GLuint a, GLuint b, int size);
void innerDNN_shaders_rmsnorm(shaderPrograms* prog, GLuint o, GLuint x, GLuint weight, int size, int weight_offset, GLuint cache_1, GLuint cache_2);
void innerDNN_shaders_softmax(shaderPrograms* prog, GLuint x, int size_x, int size_y, GLuint cache_1, GLuint cache_2, GLuint cache_3, GLuint cache_4);
void innerDNN_shaders_sigmoid(shaderPrograms* prog, GLuint x, GLuint xout, int size);


void innerDNN_shaders_transformer_softmax(shaderPrograms* prog,
                                          GLuint x,
                                          int pos,
                                          int seq_len,
                                          int n_heads,
                                          GLuint transformer_softmax_cache,
                                          GLuint cache_1,
                                          GLuint cache_2,
                                          GLuint cache_3,
                                          GLuint cache_4);
void innerDNN_shaders_transformer_sum(shaderPrograms* prog, GLuint outMat, GLuint inMat, int size_x, int size_y, GLuint cache_1, GLuint cache_2);

void innerDNN_shaders_transformer_build_attMat(shaderPrograms* prog,
                                               GLuint value_cache,
                                               GLuint att,
                                               GLuint attMat,
                                               int seq_len,
                                               int pos,
                                               int head_size,
                                               int dim,
                                               int layerIdx,
                                               int n_heads);
void innerDNN_shaders_transformer_get_query_vector(shaderPrograms* prog,
                                                   GLuint q,
                                                   GLuint key_cache,
                                                   GLuint att,
                                                   int seq_len,
                                                   int pos,
                                                   int head_size,
                                                   int dim,
                                                   int layerIdx,
                                                   int n_heads);
void innerDNN_shaders_transformer_silu_and_mulW(shaderPrograms* prog,
                                                GLuint hb,
                                                GLuint hb2,
                                                int hidden_dim_vec4);
void innerDNN_shaders_transformer_posEncoding(shaderPrograms* prog,
                                                GLuint freq_cis,
                                                GLuint q,
                                                GLuint k,
                                                int pos,
                                                int dim,
                                                int hidden_dim,
                                                int freq_cis_idx_delta,
                                                int head_size);

void innerDNN_shaders_rwkv_att_wkv(
    shaderPrograms* prog,
    GLuint att_time_first,
    GLuint att_time_decay,
    GLuint k,
    GLuint v,
    GLuint aa,
    GLuint bb,
    GLuint pp,
    GLuint wkv,
    int size);

void innerDNN_shaders_rwkv_att_rkv(
    shaderPrograms* prog,
    GLuint att_time_mix_k,
    GLuint att_time_mix_v,
    GLuint att_time_mix_r,
    GLuint x,
    GLuint x_prev,
    GLuint xr,
    GLuint xk,
    GLuint xv,
    int size);

void innerDNN_shaders_rwkv_ffn(
    shaderPrograms* prog,
    GLuint att_time_mix_k,
    GLuint att_time_mix_r,
    GLuint x,
    GLuint x_prev,
    GLuint xr,
    GLuint xk,
    int size);

void innerDNN_shaders_rwkv_relu_and_sqr(shaderPrograms* prog,
                                        GLuint x,
                                        GLuint xout,
                                        int size);

#endif