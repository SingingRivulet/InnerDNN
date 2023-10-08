#include <stdio.h>
#include "../src/cJSON.h"
#include "innerDNN/gpu.h"

typedef struct {
    int dim;         // transformer dimension
    int hidden_dim;  // for ffn layers
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int seq_len;     // max sequence length
} Config;

int main() {
    Config config;
    int fd = 0;          // file descriptor for memory mapping
    float* data = NULL;  // memory mapped data pointer
    ssize_t file_size;   // size of the checkpoint file in bytes
    const char* checkpoint = "../../res/llama2/stories15M.bin";
    {
        FILE* file = fopen(checkpoint, "rb");
        if (!file) {
            fprintf(stderr, "Couldn't open file %s\n", checkpoint);
            return 1;
        }
        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) {
            return 1;
        }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        // int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END);  // move file pointer to end of file
        file_size = ftell(file);   // get the file size, in bytes
        fclose(file);
        // memory map the Transformer weights into the data pointer
        fd = open(checkpoint, O_RDONLY);  // open in read only mode
        if (fd == -1) {
            fprintf(stderr, "open failed!\n");
            return 1;
        }
        data = (float*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            fprintf(stderr, "mmap failed!\n");
            return 1;
        }
        // float* weights_ptr = data + sizeof(Config) / sizeof(float);
    }

    // read in the tokenizer.bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    float* vocab_scores = (float*)malloc(config.vocab_size * sizeof(float));
    unsigned int max_token_length;
    {
        FILE* file = fopen("../../res/llama2/tokenizer.bin", "rb");
        if (!file) {
            fprintf(stderr, "couldn't load tokenizer.bin\n");
            return 1;
        }
        if (fread(&max_token_length, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            return 1;
        }
        int len;
        for (int i = 0; i < config.vocab_size; i++) {
            if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) {
                fprintf(stderr, "failed read\n");
                return 1;
            }
            if (fread(&len, sizeof(int), 1, file) != 1) {
                fprintf(stderr, "failed read\n");
                return 1;
            }
            vocab[i] = (char*)malloc(len + 1);
            if (fread(vocab[i], len, 1, file) != 1) {
                fprintf(stderr, "failed read\n");
                return 1;
            }
            vocab[i][len] = '\0';  // add the string terminating token
        }
        fclose(file);
    }

    FILE* ofp = fopen("../../res/llama2/vocab.json", "w");
    if (ofp) {
        for (int i = 0; i < config.vocab_size; i++) {
            cJSON* root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "str", vocab[i]);
            cJSON_AddNumberToObject(root, "score", vocab_scores[i]);
            char* json_string = cJSON_PrintUnformatted(root);
            cJSON_Delete(root);
            fprintf(ofp, "%s\n", json_string);
            free(json_string);
        }
        fclose(ofp);
    } else {
        printf("fail to write vocab\n");
    }

    for (int i = 0; i < config.vocab_size; i++) {
        free(vocab[i]);
    }
    free(vocab);
    free(vocab_scores);
    return 0;
}