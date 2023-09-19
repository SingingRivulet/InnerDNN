#include "innerDNN/shaders.h"
#include "shaders_def.h"

void innerDNN_shaders_createProgram(shaderPrograms* program) {
    program->shader_rmsnorm_squares_and_sum = innerDNN_shaders_createComputeProgram(shader_rmsnorm_squares_and_sum);
    GPU_CHECK();
    program->shader_sum = innerDNN_shaders_createComputeProgram(shader_sum);
    GPU_CHECK();
    program->shader_sum_vec4 = innerDNN_shaders_createComputeProgram(shader_sum_vec4);
    GPU_CHECK();
    program->shader_rmsnorm_normalize_and_scale = innerDNN_shaders_createComputeProgram(shader_rmsnorm_normalize_and_scale);
    GPU_CHECK();
    program->shader_rmsnorm_normalize_and_scale_inplace = innerDNN_shaders_createComputeProgram(shader_rmsnorm_normalize_and_scale_inplace);
    GPU_CHECK();
    program->shader_accum = innerDNN_shaders_createComputeProgram(shader_accum);
    GPU_CHECK();
    program->shader_positionalEncoding = innerDNN_shaders_createComputeProgram(shader_positionalEncoding);
    GPU_CHECK();
    program->shader_max = innerDNN_shaders_createComputeProgram(shader_max);
    GPU_CHECK();
    program->shader_max_vec4 = innerDNN_shaders_createComputeProgram(shader_max_vec4);
    GPU_CHECK();
    program->shader_softmax_exp = innerDNN_shaders_createComputeProgram(shader_softmax_exp);
    GPU_CHECK();
    program->shader_softmax_normalize = innerDNN_shaders_createComputeProgram(shader_softmax_normalize);
    GPU_CHECK();
    program->shader_transformer_get_query_vector = innerDNN_shaders_createComputeProgram(shader_transformer_get_query_vector);
    GPU_CHECK();
    program->shader_transformer_silu_and_mulW3 = innerDNN_shaders_createComputeProgram(shader_transformer_silu_and_mulW3);
    GPU_CHECK();
    program->shader_transformer_build_attMat = innerDNN_shaders_createComputeProgram(shader_transformer_build_attMat);
    GPU_CHECK();
    program->shader_transformer_softmax_input = innerDNN_shaders_createComputeProgram(shader_transformer_softmax_input);
    GPU_CHECK();
    program->shader_transformer_softmax_output = innerDNN_shaders_createComputeProgram(shader_transformer_softmax_output);
    GPU_CHECK();
    program->shader_temperature = innerDNN_shaders_createComputeProgram(shader_temperature);
    GPU_CHECK();
    program->shader_copyBuffer = innerDNN_shaders_createComputeProgram(shader_copyBuffer);
    GPU_CHECK();
    program->shader_matmul_trans_vec4 = innerDNN_shaders_createComputeProgram(shader_matmul_trans_vec4);
    GPU_CHECK();
    program->shader_matmul = innerDNN_shaders_createComputeProgram(shader_matmul);
    GPU_CHECK();
    program->shader_sigmoid = innerDNN_shaders_createComputeProgram(shader_sigmoid);
    GPU_CHECK();
    program->shader_reluAndsqr = innerDNN_shaders_createComputeProgram(shader_reluAndsqr);
    GPU_CHECK();
    program->shader_variance_before_sum = innerDNN_shaders_createComputeProgram(shader_variance_before_sum);
    GPU_CHECK();
    program->shader_rwkv_att_rkv = innerDNN_shaders_createComputeProgram(shader_rwkv_att_rkv);
    GPU_CHECK();
    program->shader_rwkv_att_wkv = innerDNN_shaders_createComputeProgram(shader_rwkv_att_wkv);
    GPU_CHECK();
    program->shader_rwkv_ffn = innerDNN_shaders_createComputeProgram(shader_rwkv_ffn);
    GPU_CHECK();
    program->shader_layerNorm_inplace = innerDNN_shaders_createComputeProgram(shader_layerNorm_inplace);
    GPU_CHECK();
    program->shader_layerNorm = innerDNN_shaders_createComputeProgram(shader_layerNorm);
    GPU_CHECK();
}

void innerDNN_shaders_deleteProgram(shaderPrograms* prog) {
    glDeleteProgram(prog->shader_rmsnorm_squares_and_sum);
    glDeleteProgram(prog->shader_sum);
    glDeleteProgram(prog->shader_sum_vec4);
    glDeleteProgram(prog->shader_rmsnorm_normalize_and_scale);
    glDeleteProgram(prog->shader_rmsnorm_normalize_and_scale_inplace);
    glDeleteProgram(prog->shader_accum);
    glDeleteProgram(prog->shader_positionalEncoding);
    glDeleteProgram(prog->shader_max);
    glDeleteProgram(prog->shader_max_vec4);
    glDeleteProgram(prog->shader_softmax_exp);
    glDeleteProgram(prog->shader_softmax_normalize);
    glDeleteProgram(prog->shader_transformer_silu_and_mulW3);
    glDeleteProgram(prog->shader_transformer_get_query_vector);
    glDeleteProgram(prog->shader_transformer_build_attMat);
    glDeleteProgram(prog->shader_transformer_softmax_input);
    glDeleteProgram(prog->shader_transformer_softmax_output);
    glDeleteProgram(prog->shader_temperature);
    glDeleteProgram(prog->shader_copyBuffer);
    glDeleteProgram(prog->shader_matmul);
    glDeleteProgram(prog->shader_matmul_trans_vec4);
    glDeleteProgram(prog->shader_sigmoid);
    glDeleteProgram(prog->shader_reluAndsqr);
    glDeleteProgram(prog->shader_variance_before_sum);
    glDeleteProgram(prog->shader_rwkv_att_rkv);
    glDeleteProgram(prog->shader_rwkv_att_wkv);
    glDeleteProgram(prog->shader_rwkv_ffn);
    glDeleteProgram(prog->shader_layerNorm_inplace);
    glDeleteProgram(prog->shader_layerNorm);
}

// 归约法
void innerDNN_shaders_reduce_step(
    GLuint kernel,
    GLuint inBuffer,
    int insize,
    GLuint outBuffer,
    int outsize,
    int numSeq) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outBuffer);
    glUseProgram(kernel);

    int insize_gpu = glGetUniformLocation(kernel, "insize");
    glUniform1i(insize_gpu, insize);

    int shape0_gpu = glGetUniformLocation(kernel, "shape0");
    glUniform1i(shape0_gpu, outsize);

    glDispatchCompute(outsize, numSeq, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}
GLuint innerDNN_shaders_reduce_iteration(
    GLuint kernel_step,
    GLuint kernel_step_v4,
    GLuint data,
    GLuint cache_1,
    int insize,
    int numSeq,
    GLuint* otherBuffer,
    GLuint* outputAt) {
    int currentStepSize = 0;
    int nextStepSize = insize;

    GLuint currentBuffer = cache_1;
    GLuint nextBuffer = data;
    GLuint tmp;

    while (nextStepSize >= 8) {
        tmp = currentBuffer;
        currentBuffer = nextBuffer;
        nextBuffer = tmp;

        currentStepSize = nextStepSize;
        if (currentStepSize % 4 != 0) {
            break;
        }
        nextStepSize = currentStepSize / 4;
        if (nextStepSize % 4 != 0 &&                      // currentStepSize一定是4的倍数，让nextStepSize也是4的倍数，保证迭代能进行
            nextStepSize > 2) {                           // nextStepSize为2时，此次迭代后将结束循环
            nextStepSize = ((nextStepSize / 4) + 1) * 4;  // 补全到4的倍数
        }
        innerDNN_shaders_reduce_step(kernel_step_v4, currentBuffer, currentStepSize / 4, nextBuffer, nextStepSize, numSeq);
    }

    while (nextStepSize != 1) {
        // swap current and next
        tmp = currentBuffer;
        currentBuffer = nextBuffer;
        nextBuffer = tmp;

        currentStepSize = nextStepSize;
        nextStepSize = currentStepSize / 2;
        if (currentStepSize % 2 == 1) {
            nextStepSize += 1;
        }

        if (nextStepSize == 1 && outputAt != NULL) {
            nextBuffer = *outputAt;
        }
        innerDNN_shaders_reduce_step(kernel_step, currentBuffer, currentStepSize, nextBuffer, nextStepSize, numSeq);
    }
    if (otherBuffer != NULL) {
        *otherBuffer = currentBuffer;
    }
    return nextBuffer;
}
GLuint innerDNN_shaders_reduce_iteration_input(
    GLuint kernel_step,
    GLuint kernel_step_v4,
    GLuint kernel_step_input,
    GLuint data,
    GLuint cache_1,
    GLuint cache_2,
    int insize,
    int numSeq,
    GLuint* otherBuffer,
    GLuint* outputAt) {
    int currentStepSize = insize;
    int nextStepSize = currentStepSize / 2;
    if (currentStepSize % 2 == 1) {
        nextStepSize += 1;
    }

    if (nextStepSize == 1) {
        GLuint outBuffer = cache_1;
        if (outputAt != NULL) {
            outBuffer = *outputAt;
        }
        innerDNN_shaders_reduce_step(kernel_step, data, currentStepSize, outBuffer, nextStepSize, numSeq);
        if (otherBuffer != NULL) {
            *otherBuffer = cache_2;
        }
        return outBuffer;
    } else {
        int nextStepSize_v4 = nextStepSize;
        if (nextStepSize_v4 % 4 != 0 && nextStepSize > 8) {
            nextStepSize_v4 = ((nextStepSize_v4 / 4) + 1) * 4;  // 补全到4的倍数
        }
        innerDNN_shaders_reduce_step(kernel_step_input, data, currentStepSize, cache_1, nextStepSize_v4, numSeq);
        return innerDNN_shaders_reduce_iteration(kernel_step, kernel_step_v4, cache_1, cache_2, nextStepSize_v4, numSeq, otherBuffer, outputAt);
    }
}

// 矩阵与向量之间的乘法
void innerDNN_shaders_matxvec(
    shaderPrograms* prog,
    GLuint xout,
    GLuint x,
    GLuint w,
    int n,
    int d,
    int x_offset,
    int w_offset) {  // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, w);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xout);
    glUseProgram(prog->shader_matmul);

    int n_gpu = glGetUniformLocation(prog->shader_matmul, "n");
    glUniform1i(n_gpu, n);

    int x_offset_gpu = glGetUniformLocation(prog->shader_matmul, "x_offset");
    glUniform1i(x_offset_gpu, x_offset);

    int w_offset_gpu = glGetUniformLocation(prog->shader_matmul, "w_offset");
    glUniform1i(w_offset_gpu, w_offset);

    glDispatchCompute(d, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}
void innerDNN_shaders_matxvec_trans_vec4(
    shaderPrograms* prog,
    GLuint xout,
    GLuint x,
    GLuint w,
    int n,
    int d,
    int x_offset,
    int w_offset) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, w);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xout);
    glUseProgram(prog->shader_matmul_trans_vec4);

    int n_gpu = glGetUniformLocation(prog->shader_matmul_trans_vec4, "n");
    glUniform1i(n_gpu, n);

    int d_gpu = glGetUniformLocation(prog->shader_matmul_trans_vec4, "d");
    glUniform1i(d_gpu, d);

    int x_offset_gpu = glGetUniformLocation(prog->shader_matmul_trans_vec4, "x_offset");
    glUniform1i(x_offset_gpu, x_offset);

    int w_offset_gpu = glGetUniformLocation(prog->shader_matmul_trans_vec4, "w_offset");
    glUniform1i(w_offset_gpu, w_offset);

    glDispatchCompute(n / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_accum(shaderPrograms* prog, GLuint a, GLuint b, int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b);
    glUseProgram(prog->shader_accum);

    glDispatchCompute(size / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_layerNorm(shaderPrograms* prog, GLuint o, GLuint x, GLuint weight, GLuint bias, int size, int weight_offset, GLuint cache_1, GLuint cache_2, GLuint cache_3) {
    // LayerNorm is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // sum
    GLuint currentBuffer;
    GLuint nextBuffer = cache_3;
    GLuint resBuffer_sum = innerDNN_shaders_reduce_iteration_input(
        prog->shader_sum, prog->shader_sum_vec4, prog->shader_sum,
        x, cache_1, cache_2, size, 1, &currentBuffer, NULL);

    // variance
    int currentStepSize = size;
    int nextStepSize = currentStepSize / 2;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, resBuffer_sum);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, nextBuffer);
    glUseProgram(prog->shader_variance_before_sum);

    int insize = glGetUniformLocation(prog->shader_variance_before_sum, "insize");
    glUniform1i(insize, currentStepSize);

    int shape0_gpu = glGetUniformLocation(prog->shader_variance_before_sum, "shape0");
    glUniform1i(shape0_gpu, nextStepSize);

    if (nextStepSize % 4 != 0 && nextStepSize > 8) {
        nextStepSize = ((nextStepSize / 4) + 1) * 4;  // 补全到4的倍数
    }

    glDispatchCompute(nextStepSize, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();

    GLuint resBuffer_varisum = innerDNN_shaders_reduce_iteration(
        prog->shader_sum, prog->shader_sum_vec4,
        nextBuffer, currentBuffer, nextStepSize, 1, &currentBuffer, NULL);

    // layerNorm
    int weight_offset_p;
    if (o == x) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, resBuffer_sum);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resBuffer_varisum);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, weight);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bias);
        glUseProgram(prog->shader_layerNorm_inplace);

        weight_offset_p = glGetUniformLocation(prog->shader_layerNorm_inplace, "weight_offset");
        glUniform1i(weight_offset_p, weight_offset);

        glDispatchCompute(size, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        GPU_CHECK();
    } else {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, resBuffer_sum);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resBuffer_varisum);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, weight);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bias);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, o);
        glUseProgram(prog->shader_layerNorm);

        weight_offset_p = glGetUniformLocation(prog->shader_layerNorm, "weight_offset");
        glUniform1i(weight_offset_p, weight_offset);

        glDispatchCompute(size, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        GPU_CHECK();
    }
}

void innerDNN_shaders_rmsnorm(shaderPrograms* prog, GLuint o, GLuint x, GLuint weight, int size, int weight_offset, GLuint cache_1, GLuint cache_2) {
    int currentStepSize = size;
    int nextStepSize = currentStepSize / 2;

    GLuint currentBuffer = cache_1;
    GLuint nextBuffer = cache_2;
    GLuint tmp;

    if (currentStepSize % 2 == 1) {
        nextStepSize += 1;
    }

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, nextBuffer);
    glUseProgram(prog->shader_rmsnorm_squares_and_sum);

    int insize = glGetUniformLocation(prog->shader_rmsnorm_squares_and_sum, "insize");
    glUniform1i(insize, currentStepSize);

    int shape0_gpu = glGetUniformLocation(prog->shader_rmsnorm_squares_and_sum, "shape0");
    glUniform1i(shape0_gpu, nextStepSize);

    if (nextStepSize % 4 != 0 && nextStepSize > 8) {
        nextStepSize = ((nextStepSize / 4) + 1) * 4;  // 补全到4的倍数
    }

    glDispatchCompute(nextStepSize, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();

    nextBuffer = innerDNN_shaders_reduce_iteration(
        prog->shader_sum, prog->shader_sum_vec4,
        nextBuffer, currentBuffer, nextStepSize, 1, &currentBuffer, NULL);

    if (o == x) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, nextBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, weight);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, x);
        glUseProgram(prog->shader_rmsnorm_normalize_and_scale_inplace);

        int size_p = glGetUniformLocation(prog->shader_rmsnorm_normalize_and_scale_inplace, "size");
        glUniform1i(size_p, size);
        int weight_offset_p = glGetUniformLocation(prog->shader_rmsnorm_normalize_and_scale_inplace, "weight_offset");
        glUniform1i(weight_offset_p, weight_offset);

        glDispatchCompute(size, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        GPU_CHECK();

    } else {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, nextBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, weight);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, x);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, o);
        glUseProgram(prog->shader_rmsnorm_normalize_and_scale);

        int size_p = glGetUniformLocation(prog->shader_rmsnorm_normalize_and_scale, "size");
        glUniform1i(size_p, size);
        int weight_offset_p = glGetUniformLocation(prog->shader_rmsnorm_normalize_and_scale, "weight_offset");
        glUniform1i(weight_offset_p, weight_offset);

        glDispatchCompute(size, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        GPU_CHECK();
    }
}

void innerDNN_shaders_softmax(shaderPrograms* prog, GLuint x, int size_x, int size_y, GLuint cache_1, GLuint cache_2, GLuint cache_3, GLuint cache_4) {
    // find max value (for numerical stability)
    GLuint currentBuffer = cache_1;
    GLuint nextBuffer = cache_2;
    GLuint resBuffer_max;
    GLuint resBuffer_sum;

    resBuffer_max = innerDNN_shaders_reduce_iteration_input(
        prog->shader_max, prog->shader_max_vec4, prog->shader_max,
        x, cache_1, cache_2, size_x, size_y, &currentBuffer, NULL);

    // exp
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, resBuffer_max);
    glUseProgram(prog->shader_softmax_exp);

    glDispatchCompute(size_x * size_y, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();

    // sum
    resBuffer_sum = innerDNN_shaders_reduce_iteration_input(
        prog->shader_sum, prog->shader_sum_vec4, prog->shader_sum,
        x, cache_3, cache_4, size_x, size_y, &currentBuffer, NULL);

    // normalize
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, resBuffer_sum);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, resBuffer_max);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, x);
    glUseProgram(prog->shader_softmax_normalize);

    GLuint shape0 = glGetUniformLocation(prog->shader_softmax_normalize, "shape0");
    glUniform1i(shape0, size_x);

    glDispatchCompute(size_x, size_y, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_transformer_softmax(shaderPrograms* prog,
                                          GLuint x,
                                          int pos,
                                          int seq_len,
                                          int n_heads,
                                          GLuint transformer_softmax_cache,
                                          GLuint cache_1,
                                          GLuint cache_2,
                                          GLuint cache_3,
                                          GLuint cache_4) {
    int uniformVar;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, transformer_softmax_cache);
    glUseProgram(prog->shader_transformer_softmax_input);

    uniformVar = glGetUniformLocation(prog->shader_transformer_softmax_input, "seq_len");
    glUniform1i(uniformVar, seq_len);

    uniformVar = glGetUniformLocation(prog->shader_transformer_softmax_input, "pos");
    glUniform1i(uniformVar, pos);

    glDispatchCompute(n_heads, pos + 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();

    innerDNN_shaders_softmax(prog, transformer_softmax_cache, pos + 1, n_heads, cache_1, cache_2, cache_3, cache_4);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, transformer_softmax_cache);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, x);
    glUseProgram(prog->shader_transformer_softmax_output);

    uniformVar = glGetUniformLocation(prog->shader_transformer_softmax_output, "seq_len");
    glUniform1i(uniformVar, seq_len);

    uniformVar = glGetUniformLocation(prog->shader_transformer_softmax_output, "pos");
    glUniform1i(uniformVar, pos);

    glDispatchCompute(n_heads, pos + 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_transformer_sum(shaderPrograms* prog, GLuint outMat, GLuint inMat, int size_x, int size_y, GLuint cache_1, GLuint cache_2) {
    // prog, s, s->xb, s->mulBuffer_4, pos + 1, head_size, p->n_heads
    GLuint res = outMat;
    innerDNN_shaders_reduce_iteration_input(
        prog->shader_sum, prog->shader_sum_vec4, prog->shader_sum,
        inMat, cache_1, cache_2, size_x, size_y, NULL, &res);
}

void innerDNN_shaders_copyBuffer(shaderPrograms* prog, GLuint src, GLuint dst, int src_offset, int dst_offset, int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, src);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dst);
    glUseProgram(prog->shader_copyBuffer);

    int uniformVar = glGetUniformLocation(prog->shader_copyBuffer, "src_offset");
    glUniform1i(uniformVar, src_offset);

    uniformVar = glGetUniformLocation(prog->shader_copyBuffer, "dst_offset");
    glUniform1i(uniformVar, dst_offset);

    glDispatchCompute(size, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_transformer_build_attMat(shaderPrograms* prog,
                                               GLuint value_cache,
                                               GLuint att,
                                               GLuint attMat,
                                               int seq_len,
                                               int pos,
                                               int head_size,
                                               int dim,
                                               int layerIdx,
                                               int n_heads) {
    // weighted sum of the values, store back into xb
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, value_cache);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, att);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, attMat);
    glUseProgram(prog->shader_transformer_build_attMat);

    int uniformVar = glGetUniformLocation(prog->shader_transformer_build_attMat, "seq_len");
    glUniform1i(uniformVar, seq_len);

    uniformVar = glGetUniformLocation(prog->shader_transformer_build_attMat, "pos");
    glUniform1i(uniformVar, pos);

    uniformVar = glGetUniformLocation(prog->shader_transformer_build_attMat, "head_size");
    glUniform1i(uniformVar, head_size);

    uniformVar = glGetUniformLocation(prog->shader_transformer_build_attMat, "dim");
    glUniform1i(uniformVar, dim);

    uniformVar = glGetUniformLocation(prog->shader_transformer_build_attMat, "layer_idx");
    glUniform1i(uniformVar, layerIdx);

    glDispatchCompute(n_heads, head_size, pos + 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_transformer_get_query_vector(shaderPrograms* prog,
                                                   GLuint q,
                                                   GLuint key_cache,
                                                   GLuint att,
                                                   int seq_len,
                                                   int pos,
                                                   int head_size,
                                                   int dim,
                                                   int layerIdx,
                                                   int n_heads) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, q);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, key_cache);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, att);
    glUseProgram(prog->shader_transformer_get_query_vector);

    int uniformVar = glGetUniformLocation(prog->shader_transformer_get_query_vector, "seq_len");
    glUniform1i(uniformVar, seq_len);

    uniformVar = glGetUniformLocation(prog->shader_transformer_get_query_vector, "pos");
    glUniform1i(uniformVar, pos);

    uniformVar = glGetUniformLocation(prog->shader_transformer_get_query_vector, "head_size");
    glUniform1i(uniformVar, head_size);

    uniformVar = glGetUniformLocation(prog->shader_transformer_get_query_vector, "dim");
    glUniform1i(uniformVar, dim);

    uniformVar = glGetUniformLocation(prog->shader_transformer_get_query_vector, "layer_idx");
    glUniform1i(uniformVar, layerIdx);

    glDispatchCompute(n_heads, head_size, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_transformer_silu_and_mulW(shaderPrograms* prog,
                                                GLuint hb,
                                                GLuint hb2,
                                                int hidden_dim_vec4) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, hb);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, hb2);
    glUseProgram(prog->shader_transformer_silu_and_mulW3);
    glDispatchCompute(hidden_dim_vec4 / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_rwkv_relu_and_sqr(shaderPrograms* prog,
                                        GLuint x,
                                        GLuint xout,
                                        int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, xout);
    glUseProgram(prog->shader_reluAndsqr);
    glDispatchCompute(size / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_sigmoid(shaderPrograms* prog,
                              GLuint x,
                              GLuint xout,
                              int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, xout);
    glUseProgram(prog->shader_sigmoid);
    glDispatchCompute(size / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_transformer_posEncoding(shaderPrograms* prog,
                                              GLuint freq_cis,
                                              GLuint q,
                                              GLuint k,
                                              int pos,
                                              int dim,
                                              int hidden_dim,
                                              int freq_cis_idx_delta,
                                              int head_size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, freq_cis);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, q);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k);
    glUseProgram(prog->shader_positionalEncoding);

    int uniformVar = glGetUniformLocation(prog->shader_positionalEncoding, "pos");
    glUniform1i(uniformVar, pos);

    uniformVar = glGetUniformLocation(prog->shader_positionalEncoding, "dim");
    glUniform1i(uniformVar, dim);

    uniformVar = glGetUniformLocation(prog->shader_positionalEncoding, "hidden_dim");
    glUniform1i(uniformVar, hidden_dim);

    uniformVar = glGetUniformLocation(prog->shader_positionalEncoding, "freq_cis_idx_delta");
    glUniform1i(uniformVar, freq_cis_idx_delta);

    uniformVar = glGetUniformLocation(prog->shader_positionalEncoding, "head_size");
    glUniform1i(uniformVar, head_size);

    glDispatchCompute(dim / 2, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

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
    int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, att_time_first);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, att_time_decay);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, aa);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bb);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, pp);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, wkv);
    glUseProgram(prog->shader_rwkv_att_wkv);
    glDispatchCompute(size / 2, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

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
    int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, att_time_mix_k);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, att_time_mix_v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, att_time_mix_r);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, x_prev);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, xr);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, xk);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, xv);
    glUseProgram(prog->shader_rwkv_att_rkv);
    glDispatchCompute(size / 2, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}

void innerDNN_shaders_rwkv_ffn(
    shaderPrograms* prog,
    GLuint att_time_mix_k,
    GLuint att_time_mix_r,
    GLuint x,
    GLuint x_prev,
    GLuint xr,
    GLuint xk,
    int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, att_time_mix_k);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, att_time_mix_r);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, x_prev);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, xr);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, xk);
    glUseProgram(prog->shader_rwkv_ffn);
    glDispatchCompute(size / 2, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GPU_CHECK();
}
