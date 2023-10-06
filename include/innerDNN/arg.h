#ifndef INNER_DNN_ARG
#define INNER_DNN_ARG
#include <time.h>
#include "shaders.h"

typedef struct {
    float prob;
    int index;
} innerDNN_probIndex;  // struct used when sorting probabilities during top-p sampling
long innerDNN_time_in_ms();
unsigned int innerDNN_random_u32();
float innerDNN_random_f32();  // random float32 in [0,1)
int innerDNN_argmax(innerDNN_shader_programs* prog, GLuint probabilities_gpu, int n);
int innerDNN_sample(innerDNN_shader_programs* prog, GLuint probabilities_gpu, int n);
int innerDNN_compare(const void* a, const void* b);
int innerDNN_sample_topp(innerDNN_shader_programs* prog, GLuint probabilities_gpu, int n, float topp, innerDNN_probIndex* innerDNN_probIndex);

#endif
