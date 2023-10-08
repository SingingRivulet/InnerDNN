#include "innerDNN/model_rwkv.h"

void innerDNN_model_rwkv_loadWeightsFromBuffer(
    innerDNN_model_rwkv_weights_local* weights,
    innerDNN_model_rwkv_weights_def* def,
    void* buffer,
    ssize_t bufferSize) {
    innerDNN_model_rwkv_fileData* data = (innerDNN_model_rwkv_fileData*)buffer;
    void* endPtr = ((char*)buffer) + bufferSize;  // 结束位置
    // 设置数据
    def->dim = data->header.dim;
    def->dim_hidden = data->header.dim_hidden;
    def->dim_output = data->header.dim_output;
    def->numLayer = data->header.numLayer;
    def->embedding_size = data->header.embedding_size;
    def->dim_hidden_vec4 = innerDNN_getBufferVec4(def->dim_hidden);
    def->dim_vec4 = innerDNN_getBufferVec4(def->dim);
    def->dim_output_vec4 = innerDNN_getBufferVec4(def->dim_output);
    def->ffn_key_len = def->dim_hidden_vec4 * def->dim;
    def->ffn_value_len = def->dim_hidden * def->dim_vec4;
    def->weightMat_len = def->dim * def->dim_vec4;
    weights->def = def;
    // 读取数据
    const int tensor_size = def->dim;
    const int linear_mat_size = def->dim * def->dim;
    const int hidden_linear_mat_size = def->dim * def->dim_hidden;
    const int output_linear_mat_size = def->dim * def->dim_output;
    const int embeddingTable_size = def->embedding_size * def->dim;

#define shiftPtr(dis) \
    ptr += dis;

    float* ptr = data->data;

    def->token_embedding_table = ptr;
    shiftPtr(embeddingTable_size);

    weights->att_norm_weight = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->att_norm_bias = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->att_time_first = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->att_time_decay = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->att_time_mix_k = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->att_time_mix_v = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->att_time_mix_r = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->att_output = ptr;
    shiftPtr(linear_mat_size * def->numLayer);

    weights->att_receptance = ptr;
    shiftPtr(linear_mat_size * def->numLayer);

    weights->att_key = ptr;
    shiftPtr(linear_mat_size * def->numLayer);

    weights->att_value = ptr;
    shiftPtr(linear_mat_size * def->numLayer);

    weights->ffn_time_mix_k = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->ffn_time_mix_r = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->ffn_norm_weight = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->ffn_norm_bias = ptr;
    shiftPtr(tensor_size * def->numLayer);

    weights->ffn_receptance = ptr;
    shiftPtr(linear_mat_size * def->numLayer);

    weights->ffn_key = ptr;
    shiftPtr(hidden_linear_mat_size * def->numLayer);

    weights->ffn_value = ptr;
    shiftPtr(hidden_linear_mat_size * def->numLayer);

    weights->input_weight = ptr;
    shiftPtr(tensor_size);

    weights->input_bias = ptr;
    shiftPtr(tensor_size);

    weights->output_weight = ptr;
    shiftPtr(tensor_size);

    weights->output_bias = ptr;
    shiftPtr(tensor_size);

    weights->output_head = ptr;
    shiftPtr(output_linear_mat_size);
}

void innerDNN_model_rwkv_saveWeightsToFile(
    FILE* file,
    innerDNN_model_rwkv_weights_local* weights_local) {
    // 写入文件头
    innerDNN_model_rwkv_fileData_header header;
    header.dim = weights_local->def->dim;
    header.dim_hidden = weights_local->def->dim_hidden;
    header.dim_output = weights_local->def->dim_output;
    header.embedding_size = weights_local->def->embedding_size;
    header.numLayer = weights_local->def->numLayer;
    fwrite(&header, sizeof(header), 1, file);
    // 计算权重大小
    const int tensor_size = header.dim * sizeof(float);
    const int linear_mat_size = header.dim * header.dim * sizeof(float);
    const int hidden_linear_mat_size = header.dim * header.dim_hidden * sizeof(float);
    const int output_linear_mat_size = header.dim * header.dim_output * sizeof(float);
    const int embeddingTable_size = header.embedding_size * header.dim * sizeof(float);
    // 写入权重矩阵
#define writeMat(w, size) fwrite(w, size, 1, file);
    writeMat(weights_local->def->token_embedding_table, embeddingTable_size);
    writeMat(weights_local->att_norm_weight, tensor_size * header.numLayer);
    writeMat(weights_local->att_norm_bias, tensor_size * header.numLayer);
    writeMat(weights_local->att_time_first, tensor_size * header.numLayer);
    writeMat(weights_local->att_time_decay, tensor_size * header.numLayer);
    writeMat(weights_local->att_time_mix_k, tensor_size * header.numLayer);
    writeMat(weights_local->att_time_mix_v, tensor_size * header.numLayer);
    writeMat(weights_local->att_time_mix_r, tensor_size * header.numLayer);
    writeMat(weights_local->att_output, linear_mat_size * header.numLayer);
    writeMat(weights_local->att_receptance, linear_mat_size * header.numLayer);
    writeMat(weights_local->att_key, linear_mat_size * header.numLayer);
    writeMat(weights_local->att_value, linear_mat_size * header.numLayer);
    writeMat(weights_local->ffn_time_mix_k, tensor_size * header.numLayer);
    writeMat(weights_local->ffn_time_mix_r, tensor_size * header.numLayer);
    writeMat(weights_local->ffn_norm_weight, tensor_size * header.numLayer);
    writeMat(weights_local->ffn_norm_bias, tensor_size * header.numLayer);
    writeMat(weights_local->ffn_receptance, linear_mat_size * header.numLayer);
    writeMat(weights_local->ffn_key, hidden_linear_mat_size * header.numLayer);
    writeMat(weights_local->ffn_value, hidden_linear_mat_size * header.numLayer);
    writeMat(weights_local->input_weight, tensor_size);
    writeMat(weights_local->input_bias, tensor_size);
    writeMat(weights_local->output_weight, tensor_size);
    writeMat(weights_local->output_bias, tensor_size);
    writeMat(weights_local->output_head, output_linear_mat_size);
#undef writeMat
}

void innerDNN_model_rwkv_weights_upload(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_weights_local* weights_local) {
    // 初始化尺寸
    weights->def = weights_local->def;
    weights->def->dim_hidden_vec4 = innerDNN_getBufferVec4(weights->def->dim_hidden);
    weights->def->dim_vec4 = innerDNN_getBufferVec4(weights->def->dim);
    weights->def->dim_output_vec4 = innerDNN_getBufferVec4(weights->def->dim_output);
    weights->def->ffn_key_len = weights->def->dim_hidden_vec4 * weights->def->dim;
    weights->def->ffn_value_len = weights->def->dim_hidden * weights->def->dim_vec4;
    weights->def->weightMat_len = weights->def->dim * weights->def->dim_vec4;

    // 上传数据
    weights->att_norm_weight = innerDNN_create_GPU_tensor_vec4(weights_local->att_norm_weight, weights->def->dim, weights->def->numLayer);
    weights->att_norm_bias = innerDNN_create_GPU_tensor_vec4(weights_local->att_norm_bias, weights->def->dim, weights->def->numLayer);
    weights->att_time_first = innerDNN_create_GPU_tensor_vec4(weights_local->att_time_first, weights->def->dim, weights->def->numLayer);
    weights->att_time_decay = innerDNN_create_GPU_tensor_vec4(weights_local->att_time_decay, weights->def->dim, weights->def->numLayer);
    weights->att_time_mix_k = innerDNN_create_GPU_tensor_vec4(weights_local->att_time_mix_k, weights->def->dim, weights->def->numLayer);
    weights->att_time_mix_v = innerDNN_create_GPU_tensor_vec4(weights_local->att_time_mix_v, weights->def->dim, weights->def->numLayer);
    weights->att_time_mix_r = innerDNN_create_GPU_tensor_vec4(weights_local->att_time_mix_r, weights->def->dim, weights->def->numLayer);
    weights->att_output = innerDNN_create_GPU_weight_vec4(weights_local->att_output, weights->def->dim, weights->def->dim, weights->def->numLayer);
    weights->att_receptance = innerDNN_create_GPU_weight_vec4(weights_local->att_receptance, weights->def->dim, weights->def->dim, weights->def->numLayer);
    weights->att_key = innerDNN_create_GPU_weight_vec4(weights_local->att_key, weights->def->dim, weights->def->dim, weights->def->numLayer);
    weights->att_value = innerDNN_create_GPU_weight_vec4(weights_local->att_value, weights->def->dim, weights->def->dim, weights->def->numLayer);
    weights->ffn_time_mix_k = innerDNN_create_GPU_tensor_vec4(weights_local->ffn_time_mix_k, weights->def->dim, weights->def->numLayer);
    weights->ffn_time_mix_r = innerDNN_create_GPU_tensor_vec4(weights_local->ffn_time_mix_r, weights->def->dim, weights->def->numLayer);
    weights->ffn_norm_weight = innerDNN_create_GPU_tensor_vec4(weights_local->ffn_norm_weight, weights->def->dim, weights->def->numLayer);
    weights->ffn_norm_bias = innerDNN_create_GPU_tensor_vec4(weights_local->ffn_norm_bias, weights->def->dim, weights->def->numLayer);
    weights->ffn_receptance = innerDNN_create_GPU_weight_vec4(weights_local->ffn_receptance, weights->def->dim, weights->def->dim, weights->def->numLayer);
    weights->ffn_key = innerDNN_create_GPU_weight_vec4(weights_local->ffn_key, weights->def->dim_hidden, weights->def->dim, weights->def->numLayer);
    weights->ffn_value = innerDNN_create_GPU_weight_vec4(weights_local->ffn_value, weights->def->dim, weights->def->dim_hidden, weights->def->numLayer);

    weights->input_weight = innerDNN_create_GPU_tensor_vec4(weights_local->input_weight, weights->def->dim, 1);
    weights->input_bias = innerDNN_create_GPU_tensor_vec4(weights_local->input_bias, weights->def->dim, 1);

    weights->output_weight = innerDNN_create_GPU_tensor_vec4(weights_local->output_weight, weights->def->dim, 1);
    weights->output_bias = innerDNN_create_GPU_tensor_vec4(weights_local->output_bias, weights->def->dim, 1);
    weights->output_head = innerDNN_create_GPU_weight_vec4(weights_local->output_head, weights->def->dim_output, weights->def->dim, 1);
}

void innerDNN_model_rwkv_weights_release(innerDNN_model_rwkv_weights_gpu* weights) {
    glDeleteBuffers(1, &weights->att_norm_weight);
    glDeleteBuffers(1, &weights->att_norm_bias);
    glDeleteBuffers(1, &weights->att_time_first);
    glDeleteBuffers(1, &weights->att_time_decay);
    glDeleteBuffers(1, &weights->att_time_mix_k);
    glDeleteBuffers(1, &weights->att_time_mix_v);
    glDeleteBuffers(1, &weights->att_time_mix_r);
    glDeleteBuffers(1, &weights->att_output);
    glDeleteBuffers(1, &weights->att_receptance);
    glDeleteBuffers(1, &weights->att_key);
    glDeleteBuffers(1, &weights->att_value);
    glDeleteBuffers(1, &weights->ffn_time_mix_k);
    glDeleteBuffers(1, &weights->ffn_time_mix_r);
    glDeleteBuffers(1, &weights->ffn_norm_weight);
    glDeleteBuffers(1, &weights->ffn_norm_bias);
    glDeleteBuffers(1, &weights->ffn_receptance);
    glDeleteBuffers(1, &weights->ffn_key);
    glDeleteBuffers(1, &weights->ffn_value);
    glDeleteBuffers(1, &weights->input_weight);
    glDeleteBuffers(1, &weights->input_bias);
    glDeleteBuffers(1, &weights->output_weight);
    glDeleteBuffers(1, &weights->output_bias);
    glDeleteBuffers(1, &weights->output_head);
}

void innerDNN_model_rwkv_buffer_init(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_buffer* buffer) {
    innerDNN_create_GPU_buffer(buffer->x, weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(buffer->logit, weights->def->dim_output_vec4, GL_DYNAMIC_DRAW, NULL);
    for (int i = 0; i < 15; ++i) {
        innerDNN_create_GPU_buffer(buffer->buffer[i], weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    }
    for (int i = 0; i < 2; ++i) {
        innerDNN_create_GPU_buffer(
            buffer->buffer_hidden[i],
            weights->def->dim_hidden_vec4,
            GL_DYNAMIC_DRAW, NULL);
    }
}
void innerDNN_model_rwkv_buffer_release(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_buffer* buffer) {
    glDeleteBuffers(1, &buffer->x);
    glDeleteBuffers(1, &buffer->logit);
    for (int i = 0; i < 15; ++i) {
        glDeleteBuffers(1, &buffer->buffer[i]);
    }
    for (int i = 0; i < 2; ++i) {
        glDeleteBuffers(1, &buffer->buffer_hidden[i]);
    }
}

void innerDNN_model_rwkv_state_set0(
    innerDNN_shader_programs* prog,
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state){
    innerDNN_shaders_fillBuffer(prog, state->aa, 0, 0, weights->def->dim_vec4 * weights->def->numLayer);
    innerDNN_shaders_fillBuffer(prog, state->bb, 0, 0, weights->def->dim_vec4 * weights->def->numLayer);
    innerDNN_shaders_fillBuffer(prog, state->pp, 0, 0, weights->def->dim_vec4 * weights->def->numLayer);
    innerDNN_shaders_fillBuffer(prog, state->att_xx, 0, 0, weights->def->dim_vec4 * weights->def->numLayer);
    innerDNN_shaders_fillBuffer(prog, state->ffn_xx, 0, 0, weights->def->dim_vec4 * weights->def->numLayer);
}
void innerDNN_model_rwkv_state_init(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state) {
    innerDNN_create_GPU_buffer(state->aa, weights->def->dim_vec4*weights->def->numLayer, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->bb, weights->def->dim_vec4*weights->def->numLayer, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->pp, weights->def->dim_vec4*weights->def->numLayer, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->att_xx, weights->def->dim_vec4*weights->def->numLayer, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->ffn_xx, weights->def->dim_vec4*weights->def->numLayer, GL_DYNAMIC_DRAW, NULL);
}

void innerDNN_model_rwkv_state_download(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state,
    float * aa, float * bb, float * pp, float * att_xx, float * ffn_xx) {
    innerDNN_downloadGPUArray(aa, state->aa, 0, weights->def->dim_vec4*weights->def->numLayer);
    innerDNN_downloadGPUArray(bb, state->bb, 0, weights->def->dim_vec4*weights->def->numLayer);
    innerDNN_downloadGPUArray(pp, state->pp, 0, weights->def->dim_vec4*weights->def->numLayer);
    innerDNN_downloadGPUArray(att_xx, state->att_xx, 0, weights->def->dim_vec4*weights->def->numLayer);
    innerDNN_downloadGPUArray(ffn_xx, state->ffn_xx, 0, weights->def->dim_vec4*weights->def->numLayer);
}

void innerDNN_model_rwkv_state_upload(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state,
    float * aa, float * bb, float * pp, float * att_xx, float * ffn_xx) {
        
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, state->aa);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights->def->dim_vec4*weights->def->numLayer, aa);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, state->bb);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights->def->dim_vec4*weights->def->numLayer, bb);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, state->pp);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights->def->dim_vec4*weights->def->numLayer, pp);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, state->att_xx);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights->def->dim_vec4*weights->def->numLayer, att_xx);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, state->ffn_xx);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights->def->dim_vec4*weights->def->numLayer, ffn_xx);
    
}

void innerDNN_model_rwkv_state_release(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state) {
    glDeleteBuffers(1, &state->aa);
    glDeleteBuffers(1, &state->bb);
    glDeleteBuffers(1, &state->pp);
    glDeleteBuffers(1, &state->att_xx);
    glDeleteBuffers(1, &state->ffn_xx);
}

void innerDNN_model_rwkv_forward(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state,
    innerDNN_model_rwkv_buffer* buffer,
    innerDNN_shader_programs* prog,
    int token) {
    int i;
    innerDNN_shaders_rwkv_input(
        prog,
        buffer->x, token,
        weights->def->token_embedding_table,
        weights->input_weight, weights->input_bias,
        buffer->buffer[0], buffer->buffer[1], buffer->buffer[2],
        weights->def->dim);

    innerDNN_GPU_CHECK();

    for (i = 0; i < weights->def->numLayer; ++i) {
        innerDNN_shaders_rwkv_layer(
            prog,
            buffer->x,
            state->aa, state->bb, state->pp,
            state->att_xx, state->ffn_xx,

            weights->att_norm_weight,
            weights->att_norm_bias,
            weights->att_time_first,
            weights->att_time_decay,
            weights->att_time_mix_k,
            weights->att_time_mix_v,
            weights->att_time_mix_r,
            weights->att_output,
            weights->att_receptance,
            weights->att_key,
            weights->att_value,
            weights->ffn_time_mix_k,
            weights->ffn_time_mix_r,
            weights->ffn_norm_weight,
            weights->ffn_norm_bias,
            weights->ffn_receptance,
            weights->ffn_key,
            weights->ffn_value,

            buffer->buffer,
            buffer->buffer_hidden,

            weights->def->dim,
            weights->def->dim_hidden,
            weights->def->dim_vec4 * i,
            weights->def->weightMat_len * i,
            weights->def->ffn_key_len * i,
            weights->def->ffn_value_len * i);
        innerDNN_GPU_CHECK();
    }

    innerDNN_shaders_rwkv_output(
        prog,
        buffer->logit, buffer->x,
        weights->output_weight, weights->output_bias,
        weights->output_head,
        buffer->buffer[0], buffer->buffer[1], buffer->buffer[2], buffer->buffer[3],
        weights->def->dim, weights->def->dim_output, 0, 0);
    innerDNN_GPU_CHECK();
}
