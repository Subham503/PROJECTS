#include "InvertedIndex.h"

InvertedIndex::InvertedIndex() {
}

void InvertedIndex::addTerm(int wordID, int docID) {
    std::vector<Posting>& postings = index[wordID];
    
    // Check if the last posting is for the same document
    if (!postings.empty() && postings.back().docID == docID) {
        postings.back().frequency++;
    } else {
        // Since we process files one by one, new docIDs will always be >= previous ones
        // In a real concurrent system, we might need to search or sort, but here append is fine
        // provided we add terms for a document all at once or sequentially.
        // Actually, to be safe if terms come out of order, we should linear search the end
        // But for this simple engine, we assume file-by-file processing.
        postings.push_back({docID, 1});
    }
}

const std::vector<Posting>& InvertedIndex::getPostings(int wordID) const {
    static const std::vector<Posting> empty;
    auto it = index.find(wordID);
    if (it != index.end()) {
        return it->second;
    }
    return empty;
}
