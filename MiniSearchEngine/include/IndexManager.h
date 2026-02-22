#ifndef INDEX_MANAGER_H
#define INDEX_MANAGER_H

#include "Trie.h"
#include "InvertedIndex.h"
#include "DocumentStore.h"
#include "Tokenizer.h"
#include <memory>
#include <vector>

struct SearchResult {
    std::string documentName;
    double score;
};

class IndexManager {
public:
    IndexManager();
    
    void init(const std::string& stopwordsPath);
    void indexFile(const std::string& filepath);
    std::vector<SearchResult> search(const std::string& query);

private:
    Trie trie;
    InvertedIndex invertedIndex;
    DocumentStore docStore;
    Tokenizer tokenizer;
    
    // Internal helper for TF-IDF
    double calculateIDF(int wordID);
};

#endif // INDEX_MANAGER_H
