#ifndef TRIE_H
#define TRIE_H

#include <string>
#include <unordered_map>
#include <memory>

struct TrieNode {
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    int wordID = -1; // -1 indicates not a terminal node or not indexed
};

class Trie {
public:
    Trie();
    
    // Inserts a word and assigns a unique WordID if new.
    // Returns the WordID associated with the word.
    int insert(const std::string& word);
    
    // Searches for a word and returns its WordID.
    // Returns -1 if not found.
    int search(const std::string& word) const;

private:
    std::unique_ptr<TrieNode> root;
    int nextWordID;
};

#endif // TRIE_H
