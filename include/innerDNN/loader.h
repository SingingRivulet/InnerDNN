#ifndef INNER_DNN_LOADER
#define INNER_DNN_LOADER

#include <stdio.h>
#include "gpu.h"

typedef struct {
    int size;
    int fd;
    void* data;
} innerDNN_file2memory;
void innerDNN_loadFile(innerDNN_file2memory* f, const char* checkpoint);
void innerDNN_unloadFile(innerDNN_file2memory* f);

#endif