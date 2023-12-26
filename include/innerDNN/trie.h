#ifndef INNER_DNN_TRIE
#define INNER_DNN_TRIE

#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

struct innerDNN_trie_node_t;
struct innerDNN_trie_pair_t {
    struct innerDNN_trie_node_t* node;
    char value;
};

struct innerDNN_trie_node_t {
    bool isLeaf;
    void* value;
    struct innerDNN_trie_pair_t* children;
    int children_count;
};
typedef struct {
    struct innerDNN_trie_node_t* root;
} innerDNN_trie;

typedef struct innerDNN_trie_node_t innerDNN_trie_node;

innerDNN_trie_node* innerDNN_trie_createNode();  // 创建节点

innerDNN_trie_node* innerDNN_trie_getChildNode(  // 获取子节点
    innerDNN_trie_node* node,
    char key,
    bool createMode,  // 0:找不到就返回NULL 1:找不到就自动创建（会使排序模式失效）
    bool binSearch);  // 二分搜索

void innerDNN_trie_sortNode(innerDNN_trie_node* node);  // 对节点的子节点排序

void innerDNN_trie_freeNode(innerDNN_trie_node* node);  // 释放节点，并递归操作

void innerDNN_trie_insert(innerDNN_trie* tree, const char* key, void* value);  // 插入数据

void* innerDNN_trie_search(innerDNN_trie* tree, const char* str);  // 搜索

// 不提供删除，因为目前没必要

void innerDNN_trie_init(innerDNN_trie* self);
void innerDNN_trie_destroy(innerDNN_trie* self);

#endif
