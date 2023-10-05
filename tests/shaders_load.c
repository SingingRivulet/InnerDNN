#include "innerDNN/model_rwkv.h"
int main() {
    innerDNN_GPUContext context;        // gpu上下文
    innerDNN_shader_programs programs;  // shader
    innerDNN_create_GPUContext(&context);
    innerDNN_shaders_createProgram(&programs);
    innerDNN_shaders_deleteProgram(&programs);
    innerDNN_release_GPUContext(&context);
    return 0;
}