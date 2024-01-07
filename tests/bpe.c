#include "innerDNN/bpe.h"
void test_load() {
    innerDNN_bpe_vocab* vocab = innerDNN_bpe_loadVocabFromFile("../../res/test_vocab.json");
    innerDNN_bpe_releaseVocab(vocab);

    vocab = innerDNN_bpe_loadVocabFromFile("../../res/rwkv/vocab.json");
    innerDNN_bpe_createIndexer(vocab);
    // 显示所有字符
    //  for (int i = 0; i < vocab->count; ++i) {
    //      printf("%s %f %d\n", vocab->words[i].str, vocab->words[i].score, vocab->words[i].id);
    //  }
    const char* testStr = "hello world!innerDNN";
    int ntokens;
    innerDNN_bpe_vocab_item* tokens[32];
    int tokens_id[32];
    innerDNN_bpe_encode(testStr, vocab, 32, tokens, tokens_id, &ntokens);
    char str_buffer[128];
    for (int i = 0; i < ntokens; ++i) {
        printf("%s score=%f index=%d realId=%d\n",
               tokens[i]->str, tokens[i]->score, tokens[i]->id, tokens_id[i]);
    }

    // 解码
    innerDNN_bpe_decode(vocab, str_buffer, sizeof(str_buffer), tokens_id, ntokens);
    printf("\nbpe decode:%s\n", str_buffer);

    printf("releasing vocab...\n");
    innerDNN_bpe_releaseVocab(vocab);
    printf("success\n");
}
int main() {
    test_load();
    return 0;
}