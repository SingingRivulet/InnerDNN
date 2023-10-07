#include "innerDNN/bpe.h"
void test_load() {
    innerDNN_bpe_vocab* vocab = innerDNN_bpe_loadVocabFromFile("../../res/test_vocab.json");
    innerDNN_bpe_releaseVocab(vocab);
}
int main() {
    test_load();
    return 0;
}