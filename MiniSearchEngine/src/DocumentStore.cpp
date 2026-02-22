#include "DocumentStore.h"
#include <stdexcept>

DocumentStore::DocumentStore() : nextDocID(0) {
}

int DocumentStore::addDocument(const std::string& filepath) {
    int id = nextDocID++;
    docs[id] = {id, filepath, 0};
    return id;
}

const Document& DocumentStore::getDocument(int docID) const {
    auto it = docs.find(docID);
    if (it != docs.end()) {
        return it->second;
    }
    throw std::runtime_error("Document ID not found");
}

void DocumentStore::setWordCount(int docID, int count) {
    if (docs.find(docID) != docs.end()) {
        docs[docID].totalWordCount = count;
    }
}

int DocumentStore::getTotalDocuments() const {
    return docs.size();
}
