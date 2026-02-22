#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_set>

class Tokenizer {
public:
    void loadStopwords(const std::string& filepath);
    std::vector<std::string> tokenize(const std::string& text);
    std::string readFile(const std::string& filepath);

private:
    std::unordered_set<std::string> stopwords;
};

#endif // TOKENIZER_H
