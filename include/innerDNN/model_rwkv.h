#ifndef INNER_DNN_MODEL_RWKV
#define INNER_DNN_MODEL_RWKV
#include "shaders.h"
#include <stdint.h>

typedef struct {
    GLuint aa;
    GLuint bb;
    GLuint pp;
    GLuint att_xx;
    GLuint ffn_xx;
} innerDNN_model_rwkv_state;

typedef struct {
    float* token_embedding_table;

    int dim;
    int dim_hidden;
    int dim_output;
    int numLayer;
    int embedding_size;
    // 储存参数是对齐vec4的
    int dim_vec4;
    int dim_hidden_vec4;
    int dim_output_vec4;
    int weightMat_len;
    // 隐含层的尺寸（同样是对齐vec4）
    int ffn_key_len;
    int ffn_value_len;
} innerDNN_model_rwkv_weights_def;

typedef struct {
    float* att_norm_weight;
    float* att_norm_bias;
    float* att_time_first;
    float* att_time_decay;
    float* att_time_mix_k;
    float* att_time_mix_v;
    float* att_time_mix_r;
    float* att_output;
    float* att_receptance;
    float* att_key;
    float* att_value;
    float* ffn_time_mix_k;
    float* ffn_time_mix_r;
    float* ffn_norm_weight;
    float* ffn_norm_bias;
    float* ffn_receptance;
    float* ffn_key;
    float* ffn_value;

    float* input_weight;
    float* input_bias;

    float* output_weight;
    float* output_bias;
    float* output_head;

    innerDNN_model_rwkv_weights_def* def;
} innerDNN_model_rwkv_weights_local;

typedef struct {
    GLuint att_norm_weight;
    GLuint att_norm_bias;
    GLuint att_time_first;
    GLuint att_time_decay;
    GLuint att_time_mix_k;
    GLuint att_time_mix_v;
    GLuint att_time_mix_r;
    GLuint att_output;
    GLuint att_receptance;
    GLuint att_key;
    GLuint att_value;
    GLuint ffn_time_mix_k;
    GLuint ffn_time_mix_r;
    GLuint ffn_norm_weight;
    GLuint ffn_norm_bias;
    GLuint ffn_receptance;
    GLuint ffn_key;
    GLuint ffn_value;

    GLuint input_weight;
    GLuint input_bias;

    GLuint output_weight;
    GLuint output_bias;
    GLuint output_head;

    innerDNN_model_rwkv_weights_def* def;
} innerDNN_model_rwkv_weights_gpu;

typedef struct {
    GLuint buffer[15];
    GLuint buffer_hidden[2];
    GLuint x;
    GLuint logit;
} innerDNN_model_rwkv_buffer;

#pragma pack(push, 1)
typedef struct {
    int32_t dim;
    int32_t dim_hidden;
    int32_t dim_output;
    int32_t numLayer;
    int32_t embedding_size;
} innerDNN_model_rwkv_fileData_header;
typedef struct {
    innerDNN_model_rwkv_fileData_header header;
    float data[];
} innerDNN_model_rwkv_fileData;
#pragma pack(pop)

void innerDNN_model_rwkv_saveWeightsToFile(
    FILE* file,
    innerDNN_model_rwkv_weights_local* weights_local);

void innerDNN_model_rwkv_loadWeightsFromBuffer(
    innerDNN_model_rwkv_weights_local* weight,
    innerDNN_model_rwkv_weights_def* def,
    void* buffer,
    ssize_t bufferSize);

void innerDNN_model_rwkv_weights_upload(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_weights_local* weights_local);
void innerDNN_model_rwkv_weights_release(innerDNN_model_rwkv_weights_gpu* weights);

void innerDNN_model_rwkv_state_init(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state);
void innerDNN_model_rwkv_state_release(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state);

void innerDNN_model_rwkv_state_set0(
    innerDNN_shader_programs* prog,
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state);

void innerDNN_model_rwkv_state_download(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state,
    float* aa,
    float* bb,
    float* pp,
    float* att_xx,
    float* ffn_xx);

void innerDNN_model_rwkv_state_upload(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state,
    float* aa,
    float* bb,
    float* pp,
    float* att_xx,
    float* ffn_xx);

void innerDNN_model_rwkv_buffer_init(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_buffer* buffer);
void innerDNN_model_rwkv_buffer_release(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_buffer* buffer);

void innerDNN_model_rwkv_forward(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state,
    innerDNN_model_rwkv_buffer* buffer,
    innerDNN_shader_programs* prog,
    int token);

#endif