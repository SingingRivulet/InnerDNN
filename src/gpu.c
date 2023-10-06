#include "innerDNN/gpu.h"

void innerDNN_checkGPUError(int line) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        printf(__FILE__ ":%d glGetError returns %d\n", line, err);
        exit(1);
    }
}

void innerDNN_create_GPUContext(innerDNN_GPUContext* ctx) {
    ctx->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (ctx->display == EGL_NO_DISPLAY) {
        printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
        return;
    }

    EGLint majorVersion;
    EGLint minorVersion;
    EGLBoolean returnValue = eglInitialize(ctx->display, &majorVersion, &minorVersion);
    if (returnValue != EGL_TRUE) {
        printf("eglInitialize failed\n");
        return;
    }

    EGLConfig cfg;
    EGLint count;
    EGLint s_configAttribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_NONE};
    if (eglChooseConfig(ctx->display, s_configAttribs, &cfg, 1, &count) == EGL_FALSE) {
        printf("eglChooseConfig failed\n");
        return;
    }

    EGLint context_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    ctx->context = eglCreateContext(ctx->display, cfg, EGL_NO_CONTEXT, context_attribs);
    if (ctx->context == EGL_NO_CONTEXT) {
        printf("eglCreateContext failed\n");
        return;
    }
    returnValue = eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx->context);
    if (returnValue != EGL_TRUE) {
        printf("eglMakeCurrent failed returned %d\n", returnValue);
        return;
    }
}

void innerDNN_release_GPUContext(innerDNN_GPUContext* ctx) {
    eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(ctx->display, ctx->context);
    eglReleaseThread();
    eglTerminate(ctx->display);
}

// 加载着色器

GLuint innerDNN_shaders_loadShader(GLenum shaderType, const char* pSource) {
    GLuint shader = glCreateShader(shaderType);
    if (shader) {
        glShaderSource(shader, 1, &pSource, NULL);
        glCompileShader(shader);
        GLint compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            GLint infoLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
            if (infoLen) {
                char* buf = (char*)malloc(infoLen);
                if (buf) {
                    glGetShaderInfoLog(shader, infoLen, NULL, buf);
                    printf("%s\n\nCould not compile shader %d:\n%s\n",
                           pSource, shaderType, buf);
                    free(buf);
                    exit(1);
                }
                glDeleteShader(shader);
                shader = 0;
            }
        }
    }
    return shader;
}

GLuint innerDNN_shaders_createComputeProgram(const char* pComputeSource) {
    GLuint computeShader = innerDNN_shaders_loadShader(GL_COMPUTE_SHADER, pComputeSource);
    if (!computeShader) {
        return 0;
    }

    GLuint program = glCreateProgram();
    if (program) {
        glAttachShader(program, computeShader);
        glLinkProgram(program);
        GLint linkStatus = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        if (linkStatus != GL_TRUE) {
            GLint bufLength = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
            if (bufLength) {
                char* buf = (char*)malloc(bufLength);
                if (buf) {
                    glGetProgramInfoLog(program, bufLength, NULL, buf);
                    printf("Could not link program:\n%s\n", buf);
                    free(buf);
                    exit(1);
                }
            }
            glDeleteProgram(program);
            program = 0;
        }
    }
    return program;
}

int innerDNN_getBufferVec4(int size) {
    int n = size / 4;
    if (size % 4 != 0) {
        n += 1;
    }
    return n * 4;
}

void innerDNN_copyLocalVec(float* out, float* src, int n_layers, int dim, int dim_vec4) {
    int i, j, l;
    for (l = 0; l < n_layers; ++l) {
        for (i = 0; i < dim; ++i) {
            float val = src[l * dim + i];
            out[l * dim_vec4 + i] = val;
        }
    }
}

void innerDNN_copyLocalMat(float* out, float* src, int n_layers, int dim_i, int dim_j, int rdim) {
    int i, j, l;
    for (l = 0; l < n_layers; ++l) {
        for (i = 0; i < dim_i; ++i) {
            for (j = 0; j < dim_j; ++j) {
                float val = src[l * dim_i * dim_j + j * dim_i + i];
                out[l * rdim * dim_i + i * rdim + j] = val;
            }
            for (; j < rdim; ++j) {
                out[l * rdim * dim_i + i * rdim + j] = 0;
            }
        }
    }
}

GLuint innerDNN_create_GPU_weight(float* buffer, int len_gpu) {
    GLuint remote_w;
    innerDNN_create_GPU_buffer(remote_w, len_gpu, GL_STATIC_DRAW, buffer);
    return remote_w;
}

// 创建vec4形式的gpu矩阵
// 因为是矩阵，输入需要严格的长度
// 输出对齐vec4，多余的丢弃
GLuint innerDNN_create_GPU_weight_vec4(float* local_w, int output_dim, int input_dim, int n_layers) {
    int output_dim_vec4 = innerDNN_getBufferVec4(output_dim);
    int len_gpu = sizeof(float) * n_layers * input_dim * output_dim_vec4;
    float* tmp = (float*)malloc(len_gpu);
    GLuint remote_w;
    innerDNN_copyLocalMat(tmp, local_w, n_layers, input_dim, output_dim, output_dim_vec4);
    innerDNN_create_GPU_buffer(remote_w, len_gpu, GL_STATIC_DRAW, tmp);
    free(tmp);
    return remote_w;
}

GLuint innerDNN_create_GPU_tensor_vec4(float* local_w, int dim, int n_layers) {
    int dim_vec4 = innerDNN_getBufferVec4(dim);
    int len_gpu = sizeof(float) * n_layers * dim_vec4;
    float* tmp = (float*)malloc(len_gpu);
    GLuint remote_w;
    innerDNN_copyLocalVec(tmp, local_w, n_layers, dim, dim_vec4);
    innerDNN_create_GPU_buffer(remote_w, len_gpu, GL_STATIC_DRAW, tmp);
    free(tmp);
    return remote_w;
}
