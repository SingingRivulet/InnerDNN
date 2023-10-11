#include "innerDNN/loader.h"
void innerDNN_loadFile(innerDNN_file2memory * f, const char * checkpoint) {
    ssize_t file_size;
    FILE* file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        return;
    }
    fseek(file, 0, SEEK_END);  // move file pointer to end of file
    file_size = ftell(file);   // get the file size, in bytes
    fclose(file);
    f->size = file_size;
    printf("file size:%ld\n", file_size);
    // memory map the Transformer weights into the data pointer
    int fd = open(checkpoint, O_RDONLY);  // open in read only mode
    if (fd == -1) {
        fprintf(stderr, "open failed!\n");
        return;
    }
    void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        return;
    }
    f->fd = fd;
    f->size = file_size;
    f->data = data;
}

void innerDNN_unloadFile(innerDNN_file2memory * f) {
    if (data != MAP_FAILED)
        munmap(f->data, f->size);
    if (f->fd != -1)
        close(f->fd);
}
