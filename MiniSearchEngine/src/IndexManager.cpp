#include "IndexManager.h"
#include <cmath>
#include <algorithm>
#include <iostream>

IndexManager::IndexManager() {
}

void IndexManager::init(const std::string& stopwordsPath) {
    tokenizer.loadStopwords(stopwordsPath);
}

void IndexManager::indexFile(const std::string& filepath) {
    std::cout << "Indexing: " << filepath << std::endl;
    std::string content = tokenizer.readFile(filepath);
    if (content.empty()) return;
    
    std::vector<std::string> tokens = tokenizer.tokenize(content);
    int docID = docStore.addDocument(filepath);
    
    for (const std::string& word : tokens) {
        int wordID = trie.insert(word);
        invertedIndex.addTerm(wordID, docID);
    }
    
    docStore.setWordCount(docID, tokens.size());
}

double IndexManager::calculateIDF(int wordID) {
    int totalDocs = docStore.getTotalDocuments();
    // Get number of docs containing the term
    const auto& postings = invertedIndex.getPostings(wordID);
    int docsWithTerm = postings.size();
    
    // Use smoothed IDF: log(1 + N/n)
    if (docsWithTerm == 0) return 0.0;
    return std::log(1.0 + (double)totalDocs / (double)docsWithTerm);
}

std::vector<SearchResult> IndexManager::search(const std::string& query) {
    std::vector<std::string> queryTokens = tokenizer.tokenize(query);
    std::unordered_map<int, double> docScores;
    
    for (const std::string& word : queryTokens) {
        int wordID = trie.search(word);
        if (wordID != -1) {
            double idf = calculateIDF(wordID);
            const auto& postings = invertedIndex.getPostings(wordID);
            
            for (const auto& posting : postings) {
                const Document& doc = docStore.getDocument(posting.docID);
                double tf = (double)posting.frequency / (double)doc.totalWordCount;
                docScores[posting.docID] += tf * idf;
            }
        }
    }
    
    std::vector<SearchResult> results;
    for (const auto& pair : docScores) {
        results.push_back({docStore.getDocument(pair.first).filepath, pair.second});
    }
    
    // Sort by score descending
    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
    });
    
    return results;
}
