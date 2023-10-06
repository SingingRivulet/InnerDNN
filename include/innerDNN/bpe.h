#ifndef INNER_DNN_BPE
#define INNER_DNN_BPE
#include <string.h>
#include "gpu.h"
int innerDNN_bpe_str_lookup(char* str, char** vocab, int vocab_size);
void innerDNN_bpe_bpe_encode(char* text, char** vocab, float* vocab_scores, int vocab_size, unsigned int max_token_length, int* tokens, int* n_tokens);
#endif
