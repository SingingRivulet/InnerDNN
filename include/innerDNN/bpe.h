#ifndef INNER_DNN_BPE
#define INNER_DNN_BPE
#include <string.h>
#include "gpu.h"

typedef struct {
    int id;
    float score;
    char* str;
} innerDNN_bpe_vocab_item;

typedef struct {
    float score;
    char* str;
} innerDNN_bpe_word;

typedef struct {
    innerDNN_bpe_vocab_item* words;
    int count;
} innerDNN_bpe_vocab;

innerDNN_bpe_vocab* innerDNN_bpe_loadVocabFromFile(const char* filename);
void innerDNN_bpe_releaseVocab(innerDNN_bpe_vocab* vocab);
innerDNN_bpe_vocab_item* innerDNN_bpe_str_lookup(char* str, innerDNN_bpe_vocab* vocab);
int innerDNN_bpe_encode(char* text, innerDNN_bpe_vocab* vocab, unsigned int max_token_length, innerDNN_bpe_vocab_item** tokens, int* n_tokens);

#endif
