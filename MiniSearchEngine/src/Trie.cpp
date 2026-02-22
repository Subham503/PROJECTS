#include "Trie.h"

Trie::Trie() : root(std::make_unique<TrieNode>()), nextWordID(0) {
}

int Trie::insert(const std::string& word) {
    TrieNode* current = root.get();
    for (char ch : word) {
        if (current->children.find(ch) == current->children.end()) {
            current->children[ch] = std::make_unique<TrieNode>();
        }
        current = current->children[ch].get();
    }
    
    if (current->wordID == -1) {
        current->wordID = nextWordID++;
    }
    return current->wordID;
}

int Trie::search(const std::string& word) const {
    TrieNode* current = root.get();
    for (char ch : word) {
        if (current->children.find(ch) == current->children.end()) {
            return -1;
        }
        current = current->children.at(ch).get();
    }
    return current->wordID;
}
