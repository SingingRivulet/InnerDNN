#include "innerDNN/shaders.h"
#include "shaders_def.inc"

void innerDNN_shaders_createProgram(innerDNN_shader_programs* program) {
    program->shader_rmsnorm_squares_and_sum = innerDNN_shaders_createComputeProgram(shader_rmsnorm_squares_and_sum);
    innerDNN_GPU_CHECK();
    program->shader_sum = innerDNN_shaders_createComputeProgram(shader_sum);
    innerDNN_GPU_CHECK();
    program->shader_sum_vec4 = innerDNN_shaders_createComputeProgram(shader_sum_vec4);
    innerDNN_GPU_CHECK();
    program->shader_rmsnorm_normalize_and_scale = innerDNN_shaders_createComputeProgram(shader_rmsnorm_normalize_and_scale);
    innerDNN_GPU_CHECK();
    program->shader_rmsnorm_normalize_and_scale_inplace = innerDNN_shaders_createComputeProgram(shader_rmsnorm_normalize_and_scale_inplace);
    innerDNN_GPU_CHECK();
    program->shader_accum_vec4 = innerDNN_shaders_createComputeProgram(shader_accum_vec4);
    innerDNN_GPU_CHECK();
    program->shader_positionalEncoding = innerDNN_shaders_createComputeProgram(shader_positionalEncoding);
    innerDNN_GPU_CHECK();
    program->shader_max = innerDNN_shaders_createComputeProgram(shader_max);
    innerDNN_GPU_CHECK();
    program->shader_max_vec4 = innerDNN_shaders_createComputeProgram(shader_max_vec4);
    innerDNN_GPU_CHECK();
    program->shader_softmax_exp = innerDNN_shaders_createComputeProgram(shader_softmax_exp);
    innerDNN_GPU_CHECK();
    program->shader_softmax_normalize = innerDNN_shaders_createComputeProgram(shader_softmax_normalize);
    innerDNN_GPU_CHECK();
    program->shader_transformer_get_query_vector = innerDNN_shaders_createComputeProgram(shader_transformer_get_query_vector);
    innerDNN_GPU_CHECK();
    program->shader_transformer_silu_and_mulW3_vec4 = innerDNN_shaders_createComputeProgram(shader_transformer_silu_and_mulW3_vec4);
    innerDNN_GPU_CHECK();
    program->shader_transformer_build_attMat = innerDNN_shaders_createComputeProgram(shader_transformer_build_attMat);
    innerDNN_GPU_CHECK();
    program->shader_transformer_softmax_input = innerDNN_shaders_createComputeProgram(shader_transformer_softmax_input);
    innerDNN_GPU_CHECK();
    program->shader_transformer_softmax_output = innerDNN_shaders_createComputeProgram(shader_transformer_softmax_output);
    innerDNN_GPU_CHECK();
    program->shader_temperature = innerDNN_shaders_createComputeProgram(shader_temperature);
    innerDNN_GPU_CHECK();
    program->shader_copyBuffer = innerDNN_shaders_createComputeProgram(shader_copyBuffer);
    innerDNN_GPU_CHECK();
    program->shader_fillBuffer = innerDNN_shaders_createComputeProgram(shader_fillBuffer);
    innerDNN_GPU_CHECK();
    program->shader_matmul_trans_vec4 = innerDNN_shaders_createComputeProgram(shader_matmul_trans_vec4);
    innerDNN_GPU_CHECK();
    program->shader_matmul = innerDNN_shaders_createComputeProgram(shader_matmul);
    innerDNN_GPU_CHECK();
    program->shader_sigmoid_vec4 = innerDNN_shaders_createComputeProgram(shader_sigmoid_vec4);
    innerDNN_GPU_CHECK();
    program->shader_reluAndsqr_vec4 = innerDNN_shaders_createComputeProgram(shader_reluAndsqr_vec4);
    innerDNN_GPU_CHECK();
    program->shader_variance_before_sum = innerDNN_shaders_createComputeProgram(shader_variance_before_sum);
    innerDNN_GPU_CHECK();
    program->shader_rwkv_att_rkv_vec4 = innerDNN_shaders_createComputeProgram(shader_rwkv_att_rkv_vec4);
    innerDNN_GPU_CHECK();
    program->shader_rwkv_att_wkv_vec4 = innerDNN_shaders_createComputeProgram(shader_rwkv_att_wkv_vec4);
    innerDNN_GPU_CHECK();
    program->shader_rwkv_ffn_vec4 = innerDNN_shaders_createComputeProgram(shader_rwkv_ffn_vec4);
    innerDNN_GPU_CHECK();
    program->shader_layerNorm_inplace = innerDNN_shaders_createComputeProgram(shader_layerNorm_inplace);
    innerDNN_GPU_CHECK();
    program->shader_layerNorm = innerDNN_shaders_createComputeProgram(shader_layerNorm);
    innerDNN_GPU_CHECK();
    program->shader_vecxvec_vec4 = innerDNN_shaders_createComputeProgram(shader_vecxvec_vec4);
    innerDNN_GPU_CHECK();
    program->shader_rwkv_carry_vec4 = innerDNN_shaders_createComputeProgram(shader_rwkv_carry_vec4);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_deleteProgram(innerDNN_shader_programs* prog) {
    glDeleteProgram(prog->shader_rmsnorm_squares_and_sum);
    glDeleteProgram(prog->shader_sum);
    glDeleteProgram(prog->shader_sum_vec4);
    glDeleteProgram(prog->shader_rmsnorm_normalize_and_scale);
    glDeleteProgram(prog->shader_rmsnorm_normalize_and_scale_inplace);
    glDeleteProgram(prog->shader_accum_vec4);
    glDeleteProgram(prog->shader_positionalEncoding);
    glDeleteProgram(prog->shader_max);
    glDeleteProgram(prog->shader_max_vec4);
    glDeleteProgram(prog->shader_softmax_exp);
    glDeleteProgram(prog->shader_softmax_normalize);
    glDeleteProgram(prog->shader_transformer_silu_and_mulW3_vec4);
    glDeleteProgram(prog->shader_transformer_get_query_vector);
    glDeleteProgram(prog->shader_transformer_build_attMat);
    glDeleteProgram(prog->shader_transformer_softmax_input);
    glDeleteProgram(prog->shader_transformer_softmax_output);
    glDeleteProgram(prog->shader_temperature);
    glDeleteProgram(prog->shader_copyBuffer);
    glDeleteProgram(prog->shader_fillBuffer);
    glDeleteProgram(prog->shader_matmul);
    glDeleteProgram(prog->shader_matmul_trans_vec4);
    glDeleteProgram(prog->shader_sigmoid_vec4);
    glDeleteProgram(prog->shader_reluAndsqr_vec4);
    glDeleteProgram(prog->shader_variance_before_sum);
    glDeleteProgram(prog->shader_rwkv_att_rkv_vec4);
    glDeleteProgram(prog->shader_rwkv_att_wkv_vec4);
    glDeleteProgram(prog->shader_rwkv_ffn_vec4);
    glDeleteProgram(prog->shader_layerNorm_inplace);
    glDeleteProgram(prog->shader_layerNorm);
    glDeleteProgram(prog->shader_vecxvec_vec4);
    glDeleteProgram(prog->shader_rwkv_carry_vec4);
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
    innerDNN_GPU_CHECK();
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
        if (nextStepSize % 4 != 0 &&                              // currentStepSize一定是4的倍数，让nextStepSize也是4的倍数，保证迭代能进行
            nextStepSize > 2) {                                   // nextStepSize为2时，此次迭代后将结束循环
            nextStepSize = innerDNN_getBufferVec4(nextStepSize);  // 补全到4的倍数
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
            nextStepSize_v4 = innerDNN_getBufferVec4(nextStepSize_v4);  // 补全到4的倍数
        }
        innerDNN_shaders_reduce_step(kernel_step_input, data, currentStepSize, cache_1, nextStepSize_v4, numSeq);
        return innerDNN_shaders_reduce_iteration(kernel_step, kernel_step_v4, cache_1, cache_2, nextStepSize_v4, numSeq, otherBuffer, outputAt);
    }
}

// 矩阵与向量之间的乘法
void innerDNN_shaders_matxvec(
    innerDNN_shader_programs* prog,
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
    innerDNN_GPU_CHECK();
}
void innerDNN_shaders_matxvec_trans_vec4(
    innerDNN_shader_programs* prog,
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
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_vecxvec(innerDNN_shader_programs* prog, GLuint out, GLuint a, GLuint b, int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out);
    glUseProgram(prog->shader_vecxvec_vec4);

    glDispatchCompute(size / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_accum(innerDNN_shader_programs* prog, GLuint a, GLuint b, int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b);
    glUseProgram(prog->shader_accum_vec4);

    glDispatchCompute(size / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_layerNorm(innerDNN_shader_programs* prog, GLuint o, GLuint x, GLuint weight, GLuint bias, int size, int weight_offset, GLuint cache_1, GLuint cache_2, GLuint cache_3) {
    // LayerNorm is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // sum
    GLuint currentBuffer;
    GLuint nextBuffer = cache_3;
    GLuint resBuffer_sum = innerDNN_shaders_reduce_iteration_input(
        prog->shader_sum, prog->shader_sum_vec4, prog->shader_sum,
        x, cache_1, cache_2, size, 1, &currentBuffer, NULL);

    innerDNN_GPU_CHECK();
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
        nextStepSize = innerDNN_getBufferVec4(nextStepSize);  // 补全到4的倍数
    }

    glDispatchCompute(nextStepSize, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();

    GLuint resBuffer_varisum = innerDNN_shaders_reduce_iteration(
        prog->shader_sum, prog->shader_sum_vec4,
        nextBuffer, currentBuffer, nextStepSize, 1, &currentBuffer, NULL);

    // printf("resBuffer_varisum:");
    // innerDNN_dumpGPUArray(resBuffer_varisum, 0, 1);
    // printf("\nresBuffer_sum:");
    // innerDNN_dumpGPUArray(resBuffer_sum, 0, 1);
    // printf("\n");
    // layerNorm
    int weight_offset_p;

    // printf("weight:");
    // innerDNN_dumpGPUArray(weight, 0, 100);
    // printf("bias:");
    // innerDNN_dumpGPUArray(bias, 0, 100);

    if (o == x) {
        // printf("inplace\n");
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, resBuffer_sum);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resBuffer_varisum);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, weight);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bias);
        glUseProgram(prog->shader_layerNorm_inplace);

        weight_offset_p = glGetUniformLocation(prog->shader_layerNorm_inplace, "weight_offset");
        glUniform1i(weight_offset_p, weight_offset);

        insize = glGetUniformLocation(prog->shader_layerNorm_inplace, "insize");
        glUniform1i(insize, size);

        glDispatchCompute(size, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        innerDNN_GPU_CHECK();
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

        insize = glGetUniformLocation(prog->shader_layerNorm, "insize");
        glUniform1i(insize, size);

        glDispatchCompute(size, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        innerDNN_GPU_CHECK();
    }
}

void innerDNN_shaders_rmsnorm(innerDNN_shader_programs* prog, GLuint o, GLuint x, GLuint weight, int size, int weight_offset, GLuint cache_1, GLuint cache_2) {
    int currentStepSize = size;
    int nextStepSize = currentStepSize / 2;

    GLuint currentBuffer = cache_1;
    GLuint nextBuffer = cache_2;
    // GLuint tmp;

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
        nextStepSize = innerDNN_getBufferVec4(nextStepSize);  // 补全到4的倍数
    }

    glDispatchCompute(nextStepSize, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();

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
        innerDNN_GPU_CHECK();

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
        innerDNN_GPU_CHECK();
    }
}

void innerDNN_shaders_softmax(innerDNN_shader_programs* prog, GLuint x, int size_x, int size_y, GLuint cache_1, GLuint cache_2, GLuint cache_3, GLuint cache_4) {
    // find max value (for numerical stability)
    GLuint currentBuffer = cache_1;
    // GLuint nextBuffer = cache_2;
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
    innerDNN_GPU_CHECK();

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
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_transformer_softmax(innerDNN_shader_programs* prog,
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
    innerDNN_GPU_CHECK();

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
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_transformer_sum(innerDNN_shader_programs* prog, GLuint outMat, GLuint inMat, int size_x, int size_y, GLuint cache_1, GLuint cache_2) {
    // prog, s, s->xb, s->mulBuffer_4, pos + 1, head_size, p->n_heads
    GLuint res = outMat;
    innerDNN_shaders_reduce_iteration_input(
        prog->shader_sum, prog->shader_sum_vec4, prog->shader_sum,
        inMat, cache_1, cache_2, size_x, size_y, NULL, &res);
}

void innerDNN_shaders_copyBuffer(innerDNN_shader_programs* prog, GLuint src, GLuint dst, int src_offset, int dst_offset, int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, src);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dst);
    glUseProgram(prog->shader_copyBuffer);

    int uniformVar = glGetUniformLocation(prog->shader_copyBuffer, "src_offset");
    glUniform1i(uniformVar, src_offset);

    uniformVar = glGetUniformLocation(prog->shader_copyBuffer, "dst_offset");
    glUniform1i(uniformVar, dst_offset);

    glDispatchCompute(size, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_fillBuffer(innerDNN_shader_programs* prog, GLuint dst, float val, int offset, int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dst);
    glUseProgram(prog->shader_fillBuffer);

    int uniformVar = glGetUniformLocation(prog->shader_fillBuffer, "offset");
    glUniform1i(uniformVar, offset);

    uniformVar = glGetUniformLocation(prog->shader_fillBuffer, "idata");
    glUniform1f(uniformVar, val);

    glDispatchCompute(size, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_transformer_build_attMat(innerDNN_shader_programs* prog,
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
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_transformer_get_query_vector(innerDNN_shader_programs* prog,
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
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_transformer_silu_and_mulW(innerDNN_shader_programs* prog,
                                                GLuint hb,
                                                GLuint hb2,
                                                int hidden_dim_vec4) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, hb);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, hb2);
    glUseProgram(prog->shader_transformer_silu_and_mulW3_vec4);
    glDispatchCompute(hidden_dim_vec4 / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_rwkv_relu_and_sqr(innerDNN_shader_programs* prog,
                                        GLuint x,
                                        GLuint xout,
                                        int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, xout);
    glUseProgram(prog->shader_reluAndsqr_vec4);
    glDispatchCompute(size / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_sigmoid(innerDNN_shader_programs* prog,
                              GLuint x,
                              GLuint xout,
                              int size) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, xout);
    glUseProgram(prog->shader_sigmoid_vec4);
    glDispatchCompute(size / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_transformer_posEncoding(innerDNN_shader_programs* prog,
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
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_rwkv_carry(
    innerDNN_shader_programs* prog,
    GLuint weight,
    GLuint bias,
    GLuint x_out,
    GLuint x,
    GLuint x_prev,
    GLuint xx,
    GLuint cache_1,
    GLuint cache_2,
    GLuint cache_3,
    int size,
    int w_offset,
    int x_offset) {
    int sizev4 = innerDNN_getBufferVec4(size);

    innerDNN_shaders_layerNorm(prog, x_out, x, weight, bias, size, w_offset, cache_1, cache_2, cache_3);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x_out);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, x_prev);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, xx);
    glUseProgram(prog->shader_rwkv_carry_vec4);

    int uniformVar = glGetUniformLocation(prog->shader_rwkv_carry_vec4, "offset");
    glUniform1i(uniformVar, x_offset / 4);

    glDispatchCompute(sizev4 / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_rwkv_att(
    innerDNN_shader_programs* prog,
    GLuint output,
    GLuint x_in,
    GLuint x_prev,
    GLuint aa,
    GLuint bb,
    GLuint pp,
    GLuint xx,
    GLuint norm_weight,
    GLuint norm_bias,
    GLuint att_time_first,
    GLuint att_time_decay,
    GLuint att_time_mix_k,
    GLuint att_time_mix_v,
    GLuint att_time_mix_r,
    GLuint att_output,
    GLuint att_receptance,
    GLuint att_key,
    GLuint att_value,
    GLuint xr,
    GLuint xk,
    GLuint xv,
    GLuint r,
    GLuint k,
    GLuint v,
    GLuint wkv,
    GLuint rwkv,
    GLuint x,
    GLuint cache_r,
    GLuint cache_1,
    GLuint cache_2,
    GLuint cache_3,
    int size,
    int rkv_w_offset,
    int mix_offset) {
    int sizev4 = innerDNN_getBufferVec4(size);
    innerDNN_shaders_rwkv_carry(
        prog, norm_weight, norm_bias, x, x_in, x_prev, xx, cache_1, cache_2, cache_3,
        size, mix_offset, mix_offset);
    innerDNN_shaders_rwkv_att_rkv(
        prog, r, k, v,
        att_time_mix_k, att_time_mix_v, att_time_mix_r,
        att_receptance, att_key, att_value,
        x, x_prev,
        xr, xk, xv, cache_r, size, rkv_w_offset, mix_offset);
    innerDNN_shaders_rwkv_att_wkv(
        prog, att_time_first, att_time_decay,
        k, v, aa, bb, pp, wkv, size, mix_offset);
    innerDNN_shaders_vecxvec(prog, rwkv, r, wkv, sizev4);
    innerDNN_shaders_matxvec_trans_vec4(
        prog, output, rwkv, att_output,
        sizev4,
        size, 0, rkv_w_offset);
}

void innerDNN_shaders_rwkv_att_wkv(
    innerDNN_shader_programs* prog,
    GLuint att_time_first,
    GLuint att_time_decay,
    GLuint k,
    GLuint v,
    GLuint aa,
    GLuint bb,
    GLuint pp,
    GLuint wkv,
    int size,
    int offset) {
    int sizev4 = innerDNN_getBufferVec4(size);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, att_time_first);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, att_time_decay);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, aa);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bb);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, pp);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, wkv);
    glUseProgram(prog->shader_rwkv_att_wkv_vec4);

    int uniformVar = glGetUniformLocation(prog->shader_rwkv_att_wkv_vec4, "offset");
    glUniform1i(uniformVar, offset / 4);

    glDispatchCompute(sizev4 / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_rwkv_att_rkv(
    innerDNN_shader_programs* prog,
    GLuint r,
    GLuint k,
    GLuint v,
    GLuint att_time_mix_k,
    GLuint att_time_mix_v,
    GLuint att_time_mix_r,
    GLuint att_receptance,
    GLuint att_key,
    GLuint att_value,
    GLuint x,
    GLuint x_prev,
    GLuint xr,
    GLuint xk,
    GLuint xv,
    GLuint cache_r,
    int size,
    int w_offset,
    int mix_offset) {
    int sizev4 = innerDNN_getBufferVec4(size);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, att_time_mix_k);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, att_time_mix_v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, att_time_mix_r);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, x_prev);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, xr);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, xk);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, xv);
    glUseProgram(prog->shader_rwkv_att_rkv_vec4);

    int uniformVar = glGetUniformLocation(prog->shader_rwkv_att_rkv_vec4, "offset");
    glUniform1i(uniformVar, mix_offset / 4);

    glDispatchCompute(sizev4 / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();

    innerDNN_shaders_matxvec_trans_vec4(
        prog, cache_r, xr, att_receptance,
        sizev4,
        size, 0, w_offset);
    innerDNN_shaders_sigmoid(prog, r, cache_r, size);

    innerDNN_shaders_matxvec_trans_vec4(
        prog, k, xk, att_key,
        sizev4,
        size, 0, w_offset);

    innerDNN_shaders_matxvec_trans_vec4(
        prog, v, xv, att_value,
        sizev4,
        size, 0, w_offset);
}

void innerDNN_shaders_rwkv_ffn(
    innerDNN_shader_programs* prog,
    GLuint ffn,
    GLuint ffn_time_mix_k,
    GLuint ffn_time_mix_r,
    GLuint x_in,
    GLuint x_prev,
    GLuint xr,
    GLuint xk,
    GLuint xx,
    GLuint norm_weight,
    GLuint norm_bias,
    GLuint ffn_receptance,
    GLuint ffn_key,
    GLuint ffn_value,
    GLuint r,
    GLuint sr,
    GLuint k,
    GLuint sk,
    GLuint wvk,
    GLuint x,
    GLuint cache_1,
    GLuint cache_2,
    GLuint cache_3,
    int size,
    int hidden_size,
    int w_offset,
    int ffn_key_offset,
    int ffn_value_offset,
    int mix_offset) {
    int sizev4 = innerDNN_getBufferVec4(size);
    int hidden_size_v4 = innerDNN_getBufferVec4(hidden_size);
    innerDNN_shaders_rwkv_carry(
        prog, norm_weight, norm_bias, x, x_in, x_prev, xx, cache_1, cache_2, cache_3,
        size, mix_offset, mix_offset);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ffn_time_mix_k);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ffn_time_mix_r);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, x);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, x_prev);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, xr);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, xk);
    glUseProgram(prog->shader_rwkv_ffn_vec4);

    int uniformVar = glGetUniformLocation(prog->shader_rwkv_ffn_vec4, "offset");
    glUniform1i(uniformVar, mix_offset / 4);

    glDispatchCompute(sizev4 / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    innerDNN_GPU_CHECK();

    innerDNN_shaders_matxvec_trans_vec4(
        prog, r, xr, ffn_receptance,
        sizev4,
        size, 0, w_offset);
    innerDNN_shaders_sigmoid(prog, r, sr, sizev4);

    // 隐含层
    innerDNN_shaders_matxvec_trans_vec4(
        prog, k, xk, ffn_key,
        sizev4,
        hidden_size, 0, ffn_key_offset);
    innerDNN_shaders_rwkv_relu_and_sqr(prog, k, sk, hidden_size_v4);
    innerDNN_shaders_matxvec_trans_vec4(
        prog, wvk, sk, ffn_value,
        hidden_size_v4,
        size, 0, ffn_value_offset);

    innerDNN_shaders_vecxvec(prog, ffn, wvk, sr, sizev4);
}

void innerDNN_shaders_rwkv_input(
    innerDNN_shader_programs* prog,
    GLuint x,
    int token,
    float* token_embedding_table,
    GLuint weight,
    GLuint bias,
    GLuint cache_1,
    GLuint cache_2,
    GLuint cache_3,
    int size) {
    float* content_row = &(token_embedding_table[token * size]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, x);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size * sizeof(float), content_row);
    // printf("x:\n");
    // innerDNN_dumpGPUArray(x, 0, 100);
    innerDNN_shaders_layerNorm(prog, x, x, weight, bias, size, 0, cache_1, cache_2, cache_3);
}

void innerDNN_shaders_rwkv_output(
    innerDNN_shader_programs* prog,
    GLuint logit,
    GLuint x,
    GLuint weight,
    GLuint bias,
    GLuint head,
    GLuint x_norm,
    GLuint cache_1,
    GLuint cache_2,
    GLuint cache_3,
    int size,
    int size_output,
    int vec_offset,
    int mat_offset) {
    // int sizev4 = innerDNN_getBufferVec4(size);
    int size_output_v4 = innerDNN_getBufferVec4(size_output);
    innerDNN_shaders_layerNorm(
        prog, x_norm, x,
        weight, bias,
        size, vec_offset, cache_1, cache_2, cache_3);
    innerDNN_GPU_CHECK();
    innerDNN_shaders_matxvec_trans_vec4(
        prog, logit, x_norm, head,
        size_output_v4,
        size, 0, mat_offset);
    innerDNN_GPU_CHECK();
}

void innerDNN_shaders_rwkv_layer(
    innerDNN_shader_programs* prog,
    GLuint x,
    GLuint aa,
    GLuint bb,
    GLuint pp,
    GLuint att_xx,
    GLuint ffn_xx,
    GLuint att_norm_weight,
    GLuint att_norm_bias,
    GLuint att_time_first,
    GLuint att_time_decay,
    GLuint att_time_mix_k,
    GLuint att_time_mix_v,
    GLuint att_time_mix_r,
    GLuint att_output,
    GLuint att_receptance,
    GLuint att_key,
    GLuint att_value,
    GLuint ffn_time_mix_k,
    GLuint ffn_time_mix_r,
    GLuint ffn_norm_weight,
    GLuint ffn_norm_bias,
    GLuint ffn_receptance,
    GLuint ffn_key,
    GLuint ffn_value,
    GLuint* caches,
    GLuint* caches_hidden,
    int size,
    int hidden_size,
    int vec_offset,
    int mat_offset,
    int ffn_key_offset,
    int ffn_value_offset) {
    int sizev4 = innerDNN_getBufferVec4(size);
    innerDNN_shaders_rwkv_att(
        prog, caches[0], x, caches[1],
        aa, bb, pp, att_xx,
        att_norm_weight, att_norm_bias,
        att_time_first, att_time_decay,
        att_time_mix_k, att_time_mix_v, att_time_mix_r,
        att_output, att_receptance,
        att_key, att_value,
        caches[2], caches[3], caches[4], caches[5], caches[6], caches[7], caches[8], caches[9],
        caches[10], caches[11], caches[12], caches[13], caches[14],
        size, mat_offset, vec_offset);
    innerDNN_GPU_CHECK();
    innerDNN_shaders_accum(prog, x, caches[0], sizev4);
    innerDNN_GPU_CHECK();
    innerDNN_shaders_rwkv_ffn(
        prog, caches[0],
        ffn_time_mix_k, ffn_time_mix_r,
        x, caches[1],
        caches[2], caches[3], ffn_xx,
        ffn_norm_weight, ffn_norm_bias, ffn_receptance,
        ffn_key, ffn_value,
        caches[4], caches[5],
        caches_hidden[0], caches_hidden[1],
        caches[6], caches[7],
        caches[8], caches[9], caches[10],
        size, hidden_size, mat_offset, ffn_key_offset, ffn_value_offset, vec_offset);
    innerDNN_GPU_CHECK();
    innerDNN_shaders_accum(prog, x, caches[0], sizev4);
    innerDNN_GPU_CHECK();
}
