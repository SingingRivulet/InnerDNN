#include "innerDNN/model_rwkv.h"
#include "innerDNN/loader.h"
void run_loadModel_fromBuffer(void* inbuffer, ssize_t size) {
    innerDNN_shader_programs programs;  // shader

    innerDNN_model_rwkv_weights_local weights_local;  // 待上传的权重
    innerDNN_model_rwkv_buffer buffer;                // 临时数据
    innerDNN_model_rwkv_state state;                  // 状态（rwkv是rnn）
    innerDNN_model_rwkv_weights_gpu weights_gpu;      // 上传后的权重
    innerDNN_model_rwkv_weights_def model_def;        // 模型定义

    innerDNN_shaders_createProgram(&programs);  // 编译着色器

    innerDNN_model_rwkv_loadWeightsFromBuffer(&weights_local, &model_def, inbuffer, size);

    innerDNN_model_rwkv_weights_upload(&weights_gpu, &weights_local);  // 上传权重
    innerDNN_model_rwkv_buffer_init(&weights_gpu, &buffer);            // 构建buffer
    innerDNN_model_rwkv_state_init(&weights_gpu, &state);              // 创建状态

    innerDNN_model_rwkv_forward(&weights_gpu, &state, &buffer, &programs, 1);  // 模型推理

    // 释放gpu端
    innerDNN_model_rwkv_state_release(&weights_gpu, &state);
    innerDNN_model_rwkv_buffer_release(&weights_gpu, &buffer);
    innerDNN_model_rwkv_weights_release(&weights_gpu);

    innerDNN_shaders_deleteProgram(&programs);
}

// 加载模型：把文件映射到内存
void run_loadModel() {
    innerDNN_file2memory checkpoint;
    innerDNN_loadFile(&checkpoint, "test.innw");
    run_loadModel_fromBuffer(checkpoint.data, checkpoint.size);
    innerDNN_unloadFile(&checkpoint);
}

int main() {
    innerDNN_GPUContext context;           // gpu上下文
    innerDNN_create_GPUContext(&context);  // 初始化gpu上下文
    run_loadModel();                       // 测试加载模型
    innerDNN_release_GPUContext(&context);
    return 0;
}
