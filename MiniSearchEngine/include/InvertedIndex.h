#ifndef INVERTED_INDEX_H
#define INVERTED_INDEX_H

#include <vector>
#include <unordered_map>
#include <algorithm>

struct Posting {
    int docID;
    int frequency;
};

class InvertedIndex {
public:
    InvertedIndex();

    // Adds a term occurrence: wordID in docID
    void addTerm(int wordID, int docID);

    // Retrieves postings for a specific wordID
    const std::vector<Posting>& getPostings(int wordID) const;

private:
    // Map wordID -> List of Postings
    std::unordered_map<int, std::vector<Posting>> index;
};

#endif // INVERTED_INDEX_H
