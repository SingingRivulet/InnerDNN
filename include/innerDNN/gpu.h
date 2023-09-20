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

void checkGPUError(int line);

#ifdef DEBUG
#define GPU_CHECK() checkGPUError(__LINE__);
#else
#define GPU_CHECK()
#endif

typedef struct {
    EGLContext context;
    EGLDisplay display;
} GPUContext;

//初始化gpu
void innerDNN_create_GPUContext(GPUContext* ctx);
void innerDNN_release_GPUContext(GPUContext* ctx);

//加载shader
GLuint innerDNN_shaders_loadShader(GLenum shaderType, const char* pSource);
GLuint innerDNN_shaders_createComputeProgram(const char* pComputeSource);

#endif