#include "innerDNN/model_rwkv.h"
int main() {
    innerDNN_GPUContext context;        // gpu上下文
    innerDNN_shader_programs programs;  // shader

    innerDNN_model_rwkv_weights_local weights_local;  // 待上传的权重
    innerDNN_model_rwkv_buffer buffer;                // 临时数据
    innerDNN_model_rwkv_state state;                  // 状态（rwkv是rnn）
    innerDNN_model_rwkv_weights_gpu weights_gpu;      // 上传后的权重
    innerDNN_model_rwkv_weights_def model_def;        // 模型定义

    // 构造本地数据
    // 模型结构
    model_def.dim = 128;
    model_def.dim_hidden = 256;
    model_def.embedding_size = 256;
    model_def.dim_output = 256;
    model_def.numLayer = 4;
    // 计算权重大小
    const int tensor_size = model_def.dim * sizeof(float);
    const int linear_mat_size = model_def.dim * model_def.dim * sizeof(float);
    const int hidden_linear_mat_size = model_def.dim * model_def.dim_hidden * sizeof(float);
    const int output_linear_mat_size = model_def.dim * model_def.dim_output * sizeof(float);
    const int embeddingTable_size = model_def.embedding_size * model_def.dim * sizeof(float);
    // 创建权重矩阵
    model_def.token_embedding_table = malloc(embeddingTable_size);
    weights_local.att_norm_weight = malloc(tensor_size * model_def.numLayer);
    weights_local.att_norm_bias = malloc(tensor_size * model_def.numLayer);
    weights_local.att_time_first = malloc(tensor_size * model_def.numLayer);
    weights_local.att_time_decay = malloc(tensor_size * model_def.numLayer);
    weights_local.att_time_mix_k = malloc(tensor_size * model_def.numLayer);
    weights_local.att_time_mix_v = malloc(tensor_size * model_def.numLayer);
    weights_local.att_time_mix_r = malloc(tensor_size * model_def.numLayer);
    weights_local.att_output = malloc(linear_mat_size * model_def.numLayer);
    weights_local.att_receptance = malloc(linear_mat_size * model_def.numLayer);
    weights_local.att_key = malloc(linear_mat_size * model_def.numLayer);
    weights_local.att_value = malloc(linear_mat_size * model_def.numLayer);
    weights_local.ffn_time_mix_k = malloc(tensor_size * model_def.numLayer);
    weights_local.ffn_time_mix_r = malloc(tensor_size * model_def.numLayer);
    weights_local.ffn_norm_weight = malloc(tensor_size * model_def.numLayer);
    weights_local.ffn_norm_bias = malloc(tensor_size * model_def.numLayer);
    weights_local.ffn_receptance = malloc(linear_mat_size * model_def.numLayer);
    weights_local.ffn_key = malloc(hidden_linear_mat_size * model_def.numLayer);
    weights_local.ffn_value = malloc(hidden_linear_mat_size * model_def.numLayer);
    weights_local.input_weight = malloc(tensor_size);
    weights_local.input_bias = malloc(tensor_size);
    weights_local.output_weight = malloc(tensor_size);
    weights_local.output_bias = malloc(tensor_size);
    weights_local.output_head = malloc(output_linear_mat_size);

    // 创建运行环境
    innerDNN_create_GPUContext(&context);       // 初始化gpu上下文
    innerDNN_shaders_createProgram(&programs);  // 编译着色器

    innerDNN_model_rwkv_weights_upload(&weights_gpu, &weights_local);  // 上传权重
    innerDNN_model_rwkv_buffer_init(&weights_gpu, &buffer);            // 构建buffer
    innerDNN_model_rwkv_state_init(&weights_gpu, &state);              // 创建状态

    // 释放gpu端
    innerDNN_model_rwkv_state_release(&weights_gpu, &state);
    innerDNN_model_rwkv_buffer_release(&weights_gpu, &buffer);
    innerDNN_model_rwkv_weights_release(&weights_gpu);

    innerDNN_shaders_deleteProgram(&programs);
    innerDNN_release_GPUContext(&context);

    // 释放
    free(model_def.token_embedding_table);
    free(weights_local.att_norm_weight);
    free(weights_local.att_norm_bias);
    free(weights_local.att_time_first);
    free(weights_local.att_time_decay);
    free(weights_local.att_time_mix_k);
    free(weights_local.att_time_mix_v);
    free(weights_local.att_time_mix_r);
    free(weights_local.att_output);
    free(weights_local.att_receptance);
    free(weights_local.att_key);
    free(weights_local.att_value);
    free(weights_local.ffn_time_mix_k);
    free(weights_local.ffn_time_mix_r);
    free(weights_local.ffn_norm_weight);
    free(weights_local.ffn_norm_bias);
    free(weights_local.ffn_receptance);
    free(weights_local.ffn_key);
    free(weights_local.ffn_value);

    free(weights_local.input_weight);
    free(weights_local.input_bias);

    free(weights_local.output_weight);
    free(weights_local.output_bias);
    free(weights_local.output_head);
    return 0;
}