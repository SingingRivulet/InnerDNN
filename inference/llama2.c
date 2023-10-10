#include "innerDNN/shaders.h"
int main() {
    innerDNN_GPUContext context;           // gpu上下文
    innerDNN_create_GPUContext(&context);  // 初始化gpu上下文
    //
    innerDNN_release_GPUContext(&context);
    return 0;
}