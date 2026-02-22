#include "Tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

void Tokenizer::loadStopwords(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string word;
    while (file >> word) {
        // Stopwords file is expected to be simple list
        stopwords.insert(word);
    }
}

std::string Tokenizer::readFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string currentToken;
    
    for (char ch : text) {
        if (std::isalnum(ch)) {
            currentToken += std::tolower(ch);
        } else {
            if (!currentToken.empty()) {
                if (stopwords.find(currentToken) == stopwords.end()) {
                    tokens.push_back(currentToken);
                }
                currentToken.clear();
            }
        }
    }
    
    if (!currentToken.empty()) {
        if (stopwords.find(currentToken) == stopwords.end()) {
            tokens.push_back(currentToken);
        }
    }
    
    return tokens;
}
