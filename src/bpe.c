#include "innerDNN/bpe.h"
#include "cJSON.h"

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

// 对已排序的数据进行处理（二分法）
innerDNN_bpe_vocab_item* innerDNN_bpe_str_lookup(char* str, innerDNN_bpe_vocab* vocab) {
    if (vocab->indexer) {
        return (innerDNN_bpe_vocab_item*)innerDNN_trie_search(vocab->indexer, str);
    } else {
        // Assume the array is sorted
        int vocab_size = vocab->count;
        int left = 0;
        int right = vocab_size - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int result = strcmp(str, vocab->words[mid].str);

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
}

static int compare(const void* a, const void* b) {
    // 提取innerDNN_bpe_vocab结构体的str指针
    const char* strA = ((const innerDNN_bpe_vocab_item*)a)->str;
    const char* strB = ((const innerDNN_bpe_vocab_item*)b)->str;

    // 使用strcmp比较字符串
    return strcmp(strA, strB);
}

innerDNN_bpe_vocab* innerDNN_bpe_loadVocabFromFile(const char* filename) {
    innerDNN_bpe_vocab* vocab = (innerDNN_bpe_vocab*)malloc(sizeof(innerDNN_bpe_vocab));
    vocab->indexer = NULL;
    vocab->words = NULL;
    FILE* file = fopen(filename, "r");  // 打开文件
    if (file != NULL) {
        char buffer[4096];
        // 计算长度
        vocab->count = 0;
        while (!feof(file)) {
            memset(buffer, 0, sizeof(buffer));
            fgets(buffer, sizeof(buffer), file);
            if (strlen(buffer) > 1) {
                vocab->count++;
            }
        }
        fseek(file, 0, SEEK_SET);
        // 加载
        vocab->words = malloc(vocab->count * sizeof(innerDNN_bpe_vocab_item));
        int id = 0;
        while (!feof(file)) {
            memset(buffer, 0, sizeof(buffer));
            fgets(buffer, sizeof(buffer), file);
            if (strlen(buffer) > 1) {
                cJSON* root = cJSON_Parse(buffer);
                if (root != NULL) {
                    cJSON* score = cJSON_GetObjectItem(root, "score");
                    cJSON* str = cJSON_GetObjectItem(root, "str");

                    if (score == NULL || str == NULL ||
                        score->type != cJSON_Number || str->type != cJSON_String) {
                        fprintf(stderr, "fail to decode:%s\n", buffer);
                        exit(1);
                    }

                    vocab->words[id].id = id;
                    vocab->words[id].length = strlen(str->valuestring);
                    vocab->words[id].str = malloc(vocab->words[id].length + 1);
                    strcpy(vocab->words[id].str, str->valuestring);
                    vocab->words[id].score = score->valueint;
                    id++;

                    cJSON_Delete(root);
                } else {
                    fprintf(stderr, "fail to decode json:%s\n", buffer);
                    exit(1);
                }
            }
        }
    } else {
        fprintf(stderr, "fail to open file:%s\n", filename);
        exit(1);
    }
    if (vocab->words) {
        qsort(vocab->words, vocab->count, sizeof(innerDNN_bpe_vocab_item), compare);
    }
    vocab->idMapper = (innerDNN_bpe_vocab_item**)malloc(sizeof(innerDNN_bpe_vocab_item*) * vocab->count);
    for (int i = 0; i < vocab->count; ++i) {
        vocab->idMapper[vocab->words[i].id] = &vocab->words[i];
    }
    return vocab;
}

void innerDNN_bpe_createIndexer(innerDNN_bpe_vocab* vocab) {
    vocab->indexer = (innerDNN_trie*)malloc(sizeof(innerDNN_trie));
    innerDNN_trie_init(vocab->indexer);
    for (int i = 0; i < vocab->count; ++i) {
        innerDNN_trie_insert(vocab->indexer, vocab->words[i].str, &vocab->words[i]);
    }
}

void innerDNN_bpe_releaseVocab(innerDNN_bpe_vocab* vocab) {
    if (vocab->indexer) {
        innerDNN_trie_destroy(vocab->indexer);
        free(vocab->indexer);
    }
    if (vocab->words != NULL) {
        for (int i = 0; i < vocab->count; ++i) {
            free(vocab->words[i].str);
        }
        free(vocab->words);
    }
    if (vocab->idMapper != NULL) {
        free(vocab->idMapper);
    }
    free(vocab);
}

int innerDNN_bpe_encode(const char* text, innerDNN_bpe_vocab* vocab, unsigned int max_token_length, innerDNN_bpe_vocab_item** tokens, int* n_tokens) {
    int status = 1;
    // a temporary buffer to merge two consecutive tokens
    char* str_buffer = (char*)malloc((max_token_length * 2 + 1) * sizeof(char));  // *2 for concat, +1 for null terminator

    // first encode every individual byte in the input string
    *n_tokens = 0;  // the number of tokens
    for (const char* c = text; *c != '\0'; c++) {
        if ((*n_tokens) >= max_token_length) {
            break;
        }
        sprintf(str_buffer, "%c", *c);
        innerDNN_bpe_vocab_item* wd = innerDNN_bpe_str_lookup(str_buffer, vocab);
        if (wd == NULL) {
            fprintf(stderr, "%s not good\n", str_buffer);
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

void innerDNN_bpe_decode(innerDNN_bpe_vocab* vocab, char* result, int resultLength, int* indices, int indexCount) {
    // 遍历索引并拼接字符串
    int i = 0;
    result[0] = '\0';
    for (i = 0; i < indexCount; i++) {
        int index_mapper = indices[i];
        if (index_mapper >= 0 && index_mapper < vocab->count) {
            innerDNN_bpe_vocab_item* item = vocab->idMapper[index_mapper];
            // 检查剩余的长度是否足够，如果不够则截断字符串
            if (resultLength - strlen(result) < item->length) {
                item->str[resultLength - strlen(result) - 1] = '\0';  // 截断字符串并添加null-terminator
                break;                                                // 跳出循环，不再继续拼接
            }
            strcat(result, item->str);  // 拼接字符串
        }
    }
}
