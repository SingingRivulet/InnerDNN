#ifndef INNER_DNN_MODEL_RWKV
#define INNER_DNN_MODEL_RWKV
#include "shaders.h"

typedef struct {
    GLuint aa;
    GLuint bb;
    GLuint pp;
    GLuint att_xx;
    GLuint ffn_xx;
} innerDNN_model_rwkv_state;

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

    float* token_embedding_table;

    int dim;
    int numLayer;
    // 储存参数是对齐vec4的
    int dim_vec4;
    int weightMat_len;
} innerDNN_model_rwkv_weights;

typedef struct {
    GLuint buffer[15];
    GLuint x;
    GLuint logit;
} innerDNN_model_rwkv_buffer;

void innerDNN_model_rwkv_weights_init();
void innerDNN_model_rwkv_weights_release();

void innerDNN_model_rwkv_state_init(
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_state* state);
void innerDNN_model_rwkv_state_release(
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_state* state);

void innerDNN_model_rwkv_buffer_init(
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_buffer* buffer);
void innerDNN_model_rwkv_buffer_release(
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_buffer* buffer);

void innerDNN_model_rwkv_forward(
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_state* state,
    innerDNN_model_rwkv_buffer* buffer,
    innerDNN_shader_programs* prog,
    int token);

#endif