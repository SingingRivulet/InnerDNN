#include "innerDNN/bpe.h"

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

//对已排序的数据进行处理（二分法）
innerDNN_bpe_vocab_item* innerDNN_bpe_str_lookup(char* str, innerDNN_bpe_vocab* vocab) {  
    // Assume the array is sorted  
    int vocab_size = vocab->count;
    int left = 0;  
    int right = vocab_size - 1;  
  
    while (left <= right) {  
        int mid = left + (right - left) / 2;  
        int result = strcmp(str, vocab->words[mid]);  
  
        if (result == 0) {  
            // Perfect match found  
            return &vocab->words[mid];  
        } else if (result < 0) {  
            // str is less than vocab[mid], continue searching in the left half  
            right = mid - 1;  
        } else {  
            // str is greater than vocab[mid], continue searching in the right half  
            left = mid + 1;  
        }  
    }  
  
    // No match found  
    return NULL;  
}
  
static innerDNN_bpe_vocab_item* loadStrings(const char* filename,int *outlen) {  
    FILE* file = fopen(filename, "r");  
    if (file == NULL) {  
        printf("Cannot open file %s\n", filename);  
        return NULL;  
    }  
  
    char* line = NULL;  
    size_t len = 0;  
    ssize_t read;  
    int count = 0; // count the number of lines  
  
    // get the number of lines in the file  
    while ((read = getline(&line, &len, file)) != -1) {  
        count++;  
    }  
  
    // rewind the file to start reading lines again  
    rewind(file);  
  
    // allocate memory for the string array  
    int id = 0;
    innerDNN_bpe_vocab_item* strings = malloc(count * sizeof(innerDNN_bpe_vocab_item)); 
    *outlen = count;
  
    // read the lines and store them in the string array  
    for (int i = 0; i < count; i++) {  
        size_t size = 0;  
        while ((read = getline(&line, &size, file)) != -1) {  
            // allocate memory for the string  
            strings[i].str = malloc(strlen(line) + 1);  
            strings[i].id = id;
            ++id;
            // copy the string into the allocated memory  
            strcpy(strings[i].str, line);  
        }  
    }  
  
    // free the memory allocated for the lines in the file  
    free(line);  
  
    fclose(file);  
  
    return strings;  
}

static int compare(const void *a, const void *b) {  
    // 提取innerDNN_bpe_vocab结构体的str指针  
    const char *strA = ((const innerDNN_bpe_vocab_item *)a)->str;  
    const char *strB = ((const innerDNN_bpe_vocab_item *)b)->str;  
  
    // 使用strcmp比较字符串  
    return strcmp(strA, strB);  
} 

innerDNN_bpe_vocab* innerDNN_bpe_loadVocabFromFile(const char* filename) {  
    innerDNN_bpe_vocab* vocab = (innerDNN_bpe_vocab*)malloc(sizeof(innerDNN_bpe_vocab));
    vocab->words = loadStrings(const char* filename,&vocab->count);
    if (vocab->words){
        qsort(vocab->words, vocab->count, sizeof(innerDNN_bpe_vocab), compare);
    }
    return vocab;
}
void innerDNN_bpe_releaseVocab(innerDNN_bpe_vocab* vocab) {  
    if(vocab->words!=NULL){
        for (int i=0;i<vocab->count;++i){
            free(vocab->words[i]->str);
        }
        free(vocab->words);
    }
    free(vocab);
}

int innerDNN_bpe_encode(char* text, innerDNN_bpe_vocab* vocab, unsigned int max_token_length, innerDNN_bpe_vocab_item* tokens, int* n_tokens) {
    int status = 1;
    // a temporary buffer to merge two consecutive tokens
    char* str_buffer = (char*)malloc((max_token_length * 2 + 1) * sizeof(char));  // *2 for concat, +1 for null terminator

    // first encode every individual byte in the input string
    *n_tokens = 0;  // the number of tokens
    for (char* c = text; *c != '\0'; c++) {
        sprintf(str_buffer, "%c", *c);
        innerDNN_bpe_vocab_item* wd = innerDNN_bpe_str_lookup(str_buffer, vocab);
        if (wd == NULL) {
            fprintf(stderr, "%s not good\n",str_buffer);
            status = 0;
            goto end;
        }
        tokens[*n_tokens] = wd;
        (*n_tokens)++;
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        innerDNN_bpe_vocab_item* best_wd = NULL;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", tokens[i]->str, tokens[i + 1]->str);
            innerDNN_bpe_vocab_item* wd = innerDNN_bpe_str_lookup(str_buffer, vocab);
            if (wd != NULL && wd->score > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = wd->score;
                best_wd = wd;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;  // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_wd;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;  // token length decreased
    }

    end:
    free(str_buffer);
    return status;
}