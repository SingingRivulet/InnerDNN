#include "innerDNN/trie.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printNode(innerDNN_trie_node* node, int level) {
    if (node->isLeaf) {
        printf("%*sLeaf: %d Value: %p\n", level * 2, " ", node->isLeaf, node->value);
    }
    for (int i = 0; i < node->children_count; i++) {
        char key = node->children[i].value;
        innerDNN_trie_node* child = node->children[i].node;
        printf("%*s%c: \n", level * 2, " ", key);
        printNode(child, level + 1);
    }
}

void testInsertion(innerDNN_trie* trie, const char* key, void* value) {
    innerDNN_trie_insert(trie, key, value);
    void* res = innerDNN_trie_search(trie, key);
    if (res == NULL) {
        printf("Insertion failed for key: %s\n", key);
    } else {
        printf("Inserted and found value for key: %s\n", key);
    }
}
int main() {
    innerDNN_trie trie;
    innerDNN_trie_init(&trie);
    testInsertion(&trie, "hello", (void*)"Hello");
    testInsertion(&trie, "world", (void*)"World");
    testInsertion(&trie, "hi", (void*)"Hi");
    testInsertion(&trie, "goodbye", (void*)"Goodbye");
    testInsertion(&trie, "apple", (void*)"Apple");
    testInsertion(&trie, "banana", (void*)"Banana");
    testInsertion(&trie, "cherry", (void*)"Cherry");
    testInsertion(&trie, "dog", (void*)"Dog");
    testInsertion(&trie, "dogs", (void*)"Dogs");  // 注意这里，我们插入了一个不同的词，但只差一个's'。这可以测试你的代码是否能够正确处理这种情况。

    void* res = innerDNN_trie_search(&trie, "do");  // 测试查找不存在的词
    printf("search \"do\":%p\n", res);

    printNode(trie.root, 0);       // 打印树以检查其结构。这只是一个简单的调试方法。你可以删除它。
    innerDNN_trie_destroy(&trie);  // 释放内存。这是好的做法，即使你可能会在程序结束时自动释放所有内存。这可以确保你没有忘记释放任何东西。
    return 0;
}