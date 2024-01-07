#ifndef INNER_DNN_BPE
#define INNER_DNN_BPE
#include <string.h>
#include "gpu.h"
#include "trie.h"

typedef struct {
    int id;
    float score;
    char* str;
    int length;
} innerDNN_bpe_vocab_item;

typedef struct {
    innerDNN_bpe_vocab_item* words;
    innerDNN_bpe_vocab_item** idMapper;
    int count;
    int id_shift;

    innerDNN_trie* indexer;
} innerDNN_bpe_vocab;

innerDNN_bpe_vocab* innerDNN_bpe_loadVocabFromFile(const char* filename);
void innerDNN_bpe_releaseVocab(innerDNN_bpe_vocab* vocab);
void innerDNN_bpe_createIndexer(innerDNN_bpe_vocab* vocab);
innerDNN_bpe_vocab_item* innerDNN_bpe_str_lookup(char* str, innerDNN_bpe_vocab* vocab);
int innerDNN_bpe_encode(const char* text, innerDNN_bpe_vocab* vocab, unsigned int max_token_length, innerDNN_bpe_vocab_item** tokens, int* tokens_id, int* n_tokens);
void innerDNN_bpe_decode(innerDNN_bpe_vocab* vocab, char* result, int resultLength, int* indices, int indexCount);

#endif
