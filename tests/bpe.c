#include "innerDNN/bpe.h"
void test_load() {
    innerDNN_bpe_vocab* vocab = innerDNN_bpe_loadVocabFromFile("../../res/test_vocab.json");
    innerDNN_bpe_releaseVocab(vocab);

    vocab = innerDNN_bpe_loadVocabFromFile("../../res/llama2/vocab.json");
    innerDNN_bpe_createIndexer(vocab);
    // 显示所有字符
    //  for (int i = 0; i < vocab->count; ++i) {
    //      printf("%s %f %d\n", vocab->words[i].str, vocab->words[i].score, vocab->words[i].id);
    //  }
    const char* testStr = "hello world!";
    int ntokens;
    innerDNN_bpe_vocab_item* tokens[16];
    innerDNN_bpe_encode(testStr, vocab, 16, tokens, &ntokens);
    for (int i = 0; i < ntokens; ++i) {
        printf("%s %f %d\n", tokens[i]->str, tokens[i]->score, tokens[i]->id);
    }
    innerDNN_bpe_releaseVocab(vocab);
}
int main() {
    test_load();
    return 0;
}