#include "innerDNN/trie.h"

innerDNN_trie_node* innerDNN_trie_createNode() {  // 创建节点
    innerDNN_trie_node* newNode = (innerDNN_trie_node*)malloc(sizeof(innerDNN_trie_node));
    newNode->isLeaf = false;
    newNode->children = NULL;
    newNode->children_count = 0;
    newNode->isLeaf = false;
    return newNode;
}

innerDNN_trie_node* innerDNN_trie_getChildNode(  // 获取子节点
    innerDNN_trie_node* node,
    char key,
    bool createMode,   // 0:找不到就返回NULL 1:找不到就自动创建（会使排序模式失效）
    bool binSearch) {  // 二分搜索
    if (binSearch) {
        if (node->children_count != 0) {
            int vocab_size = node->children_count;
            int left = 0;
            int right = vocab_size - 1;

            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (node->children[mid].value == key) {
                    return node->children[mid].node;
                } else if (node->children[mid].value > key) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
    } else {
        if (node->children_count != 0) {
            for (size_t i = 0; i < node->children_count; ++i) {
                if (node->children[i].value == key) {
                    return node->children[i].node;
                }
            }
        }
    }
    // 没找到
    if (createMode) {
        if (node->children == NULL) {
            node->children_count = 1;
            node->children = (struct innerDNN_trie_pair_t*)malloc(sizeof(struct innerDNN_trie_pair_t) * node->children_count);
            innerDNN_trie_node* res = innerDNN_trie_createNode();
            node->children[0].value = key;
            node->children[0].node = res;
            return res;
        } else {
            node->children = (struct innerDNN_trie_pair_t*)realloc(node->children, sizeof(struct innerDNN_trie_pair_t) * (node->children_count + 1));
            innerDNN_trie_node* res = innerDNN_trie_createNode();
            node->children[node->children_count].value = key;
            node->children[node->children_count].node = res;
            node->children_count += 1;
            innerDNN_trie_sortNode(node);
            return res;
        }
    } else {
        return NULL;
    }
}

// 比较函数，用于 qsort
int compare(const void* a, const void* b) {
    struct innerDNN_trie_pair_t* pairA = (struct innerDNN_trie_pair_t*)a;
    struct innerDNN_trie_pair_t* pairB = (struct innerDNN_trie_pair_t*)b;
    return pairA->value - pairB->value;
}

void innerDNN_trie_sortNode(innerDNN_trie_node* node) {  // 对节点的子节点排序
    qsort(node->children,
          node->children_count,
          sizeof(struct innerDNN_trie_pair_t),
          compare);
}

void innerDNN_trie_freeNode(innerDNN_trie_node* node) {  // 释放节点，并递归操作
    if (node->children) {
        for (size_t i = 0; i < node->children_count; i++) {
            innerDNN_trie_freeNode(node->children[i].node);
        }
        free(node->children);
    }
    free(node);
}

void innerDNN_trie_insert(innerDNN_trie* tree, const char* key, void* value) {  // 插入数据
    innerDNN_trie_node* crawl = tree->root;
    const char* c = key;
    while (*c) {
        crawl = innerDNN_trie_getChildNode(crawl, *c, true, true);
        ++c;
    }
    crawl->value = value;
    crawl->isLeaf = true;
}

void* innerDNN_trie_search(innerDNN_trie* tree, const char* str) {  // 搜索
    innerDNN_trie_node* crawl = tree->root;
    const char* c = str;
    while (*c) {
        crawl = innerDNN_trie_getChildNode(crawl, *c, false, true);
        if (crawl == NULL) {
            return NULL;
        }
        ++c;
    }

    if (crawl->isLeaf) {
        return crawl->value;
    } else {
        return NULL;
    }
}

void innerDNN_trie_init(innerDNN_trie* self) {
    self->root = innerDNN_trie_createNode();
}

void innerDNN_trie_destroy(innerDNN_trie* self) {
    innerDNN_trie_freeNode(self->root);
}