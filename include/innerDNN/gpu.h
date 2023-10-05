#ifndef INNER_DNN_GPU
#define INNER_DNN_GPU
// #define DEBUG
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl32.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void innerDNN_checkGPUError(int line);

#ifdef DEBUG
#define innerDNN_GPU_CHECK() innerDNN_checkGPUError(__LINE__);
#else
#define innerDNN_GPU_CHECK()
#endif

typedef struct {
    EGLContext context;
    EGLDisplay display;
} GPUContext;

// 初始化gpu
void innerDNN_create_GPUContext(GPUContext* ctx);
void innerDNN_release_GPUContext(GPUContext* ctx);

// 加载shader
GLuint innerDNN_shaders_loadShader(GLenum shaderType, const char* pSource);
GLuint innerDNN_shaders_createComputeProgram(const char* pComputeSource);

int innerDNN_getBufferVec4(int size);

// 复制参数矩阵
void innerDNN_copyLocalMat(float* out, float* src, int n_layers, int dim_i, int dim_j, int rdim);
void innerDNN_copyLocalVec(float* out, float* src, int n_layers, int dim, int dim_vec4);

#define innerDNN_create_GPU_buffer(ptr, size, usage, data) \
    glGenBuffers(1, &ptr);                                 \
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ptr);           \
    glBufferData(GL_SHADER_STORAGE_BUFFER,                 \
                 size,                                     \
                 data, usage);                             \
    innerDNN_GPU_CHECK();

// 创建远端数据
GLuint innerDNN_create_GPU_weight(float* buffer, int len_gpu);
GLuint innerDNN_create_GPU_weight_vec4(
    // 创建vec4对齐的矩阵。
    // 由于远端数据结构的问题，该函数只能用来上传权重矩阵
    float* local_w,
    int output_dim,
    int input_dim,
    int n_layers);

GLuint innerDNN_create_GPU_tensor_vec4(float* local_w, int dim, int n_layers);

#endif