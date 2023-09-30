#include "model_rwkv.h"

void innerDNN_model_rwkv_weights_init(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_weights_local* weights_local) {
    // 初始化尺寸
    weights->def->dim_hidden_vec4 = innerDNN_getBufferVec4(weights->def->dim_hidden);
    weights->def->dim_vec4 = innerDNN_getBufferVec4(weights->def->dim);
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
    weights->output_head = innerDNN_create_GPU_weight_vec4(weights_local->output_head, weights->def->dim, weights->def->dim, 1);
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

    free(weights->def->token_embedding_table);
    weights->def->token_embedding_table = NULL;
}

void innerDNN_model_rwkv_buffer_init(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_buffer* buffer) {
    innerDNN_create_GPU_buffer(buffer->x, weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(buffer->logit, weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
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

void innerDNN_model_rwkv_state_init(
    innerDNN_model_rwkv_weights_gpu* weights,
    innerDNN_model_rwkv_state* state) {
    innerDNN_create_GPU_buffer(state->aa, weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->bb, weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->pp, weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->att_xx, weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->ffn_xx, weights->def->dim_vec4, GL_DYNAMIC_DRAW, NULL);
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
            weights->def->dim_hidden_vec4,
            weights->def->dim_vec4 * i,
            weights->def->weightMat_len * i,
            weights->def->ffn_key_len * i,
            weights->def->ffn_value_len * i);
    }

    innerDNN_shaders_rwkv_output(
        prog,
        buffer->logit, buffer->x,
        weights->output_weight, weights->output_bias,
        weights->output_head,
        buffer->buffer[0], buffer->buffer[1], buffer->buffer[2], buffer->buffer[3],
        weights->def->dim, 0, 0);
}
