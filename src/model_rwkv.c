#include "model_rwkv.h"

void innerDNN_model_rwkv_buffer_init(
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_buffer* buffer) {
    innerDNN_create_GPU_buffer(buffer->x, weights->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(buffer->logit, weights->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    for (int i = 0; i < 15; ++i) {
        innerDNN_create_GPU_buffer(buffer->buffer[i], weights->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    }
    for (int i = 0; i < 2; ++i) {
        innerDNN_create_GPU_buffer(
            buffer->buffer_hidden[i],
            weights->dim_hidden_vec4,
            GL_DYNAMIC_DRAW, NULL);
    }
}
void innerDNN_model_rwkv_buffer_release(
    innerDNN_model_rwkv_weights* weights,
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
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_state* state) {
    innerDNN_create_GPU_buffer(state->aa, weights->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->bb, weights->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->pp, weights->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->att_xx, weights->dim_vec4, GL_DYNAMIC_DRAW, NULL);
    innerDNN_create_GPU_buffer(state->ffn_xx, weights->dim_vec4, GL_DYNAMIC_DRAW, NULL);
}

void innerDNN_model_rwkv_state_release(
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_state* state) {
    glDeleteBuffers(1, &state->aa);
    glDeleteBuffers(1, &state->bb);
    glDeleteBuffers(1, &state->pp);
    glDeleteBuffers(1, &state->att_xx);
    glDeleteBuffers(1, &state->ffn_xx);
}

void innerDNN_model_rwkv_forward(
    innerDNN_model_rwkv_weights* weights,
    innerDNN_model_rwkv_state* state,
    innerDNN_model_rwkv_buffer* buffer,
    innerDNN_shader_programs* prog,
    int token) {
    int i;
    innerDNN_shaders_rwkv_input(
        prog,
        buffer->x, token,
        weights->token_embedding_table,
        weights->input_weight, weights->input_bias,
        buffer->buffer[0], buffer->buffer[1], buffer->buffer[2],
        weights->dim);

    for (i = 0; i < weights->numLayer; ++i) {
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

            weights->dim,
            weights->dim_hidden_vec4,
            weights->dim_vec4 * i,
            weights->weightMat_len * i,
            weights->ffn_key_len * i,
            weights->ffn_value_len * i);
    }

    innerDNN_shaders_rwkv_output(
        prog,
        buffer->logit, buffer->x,
        weights->output_weight, weights->output_bias,
        weights->output_head,
        buffer->buffer[0], buffer->buffer[1], buffer->buffer[2], buffer->buffer[3],
        weights->dim, 0, 0);
}
