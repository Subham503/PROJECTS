#ifndef DOCUMENT_STORE_H
#define DOCUMENT_STORE_H

#include <string>
#include <unordered_map>
#include <vector>

struct Document {
    int id;
    std::string filepath;
    int totalWordCount;
};

class DocumentStore {
public:
    DocumentStore();

    // Registers a document and returns its new DocID
    int addDocument(const std::string& filepath);

    // Get document metadata
    const Document& getDocument(int docID) const;
    
    // Update word count for a document
    void setWordCount(int docID, int count);
    
    // Get total number of documents
    int getTotalDocuments() const;

private:
    std::unordered_map<int, Document> docs;
    int nextDocID;
};

#endif // DOCUMENT_STORE_H
