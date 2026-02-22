#include "IndexManager.h"
#include <iostream>
#include <string>
#include <vector>
#include <windows.h>

// Helper to list .txt files in a directory
std::vector<std::string> getTxtFiles(const std::string& directory) {
    std::vector<std::string> files;
    std::string pattern = directory + "\\*.txt";
    WIN32_FIND_DATA findData;
    HANDLE hFind = FindFirstFile(pattern.c_str(), &findData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.push_back(directory + "\\" + findData.cFileName);
            }
        } while (FindNextFile(hFind, &findData));
        FindClose(hFind);
    }
    return files;
}

// Simple helper to check if file has .txt extension (already implicit by *.txt pattern search)
bool isTextFile(const std::string& path) {
    return path.size() >= 4 && path.substr(path.size() - 4) == ".txt";
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "   Upgraded Mini Search Engine v1.0   " << std::endl;
    std::cout << "======================================" << std::endl;

    IndexManager engine;
    
    // 1. Initialize Stopwords
    std::cout << "[*] Loading stopwords..." << std::endl;
    // Assuming run from build directory, adjusting path might be needed.
    // Try to find data relative to current dir or known absolute locations.
    // For simplicity, we assume we are at project root or know relative path.
    // Let's assume the executable is run from project root, or we try a few specific paths.
    
    std::string dataDir = "data";
    // Basic existence check using GetFileAttributes
    if (GetFileAttributes(dataDir.c_str()) == INVALID_FILE_ATTRIBUTES) {
        if (GetFileAttributes("..\\data") != INVALID_FILE_ATTRIBUTES) {
            dataDir = "..\\data";
        } else {
             std::cerr << "Error: 'data' directory not found." << std::endl;
             return 1;
        }
    }
    
    engine.init(dataDir + "\\stopwords.txt");

    // 2. Index Files
    std::cout << "[*] Indexing files in '" << dataDir << "'..." << std::endl;
    int fileCount = 0;
    std::vector<std::string> files = getTxtFiles(dataDir);
    
    for (const auto& path : files) {
        // Skip stopwords.txt if picked up (though *.txt picks it up)
        if (path.find("stopwords.txt") == std::string::npos) {
            engine.indexFile(path);
            fileCount++;
        }
    }
    std::cout << "[*] Indexed " << fileCount << " files." << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    // 3. Search Loop
    std::string query;
    while (true) {
        std::cout << "\nEnter search query (or 'exit' to quit): ";
        std::getline(std::cin, query);
        
        if (query == "exit" || query == "quit") {
             break;
        }
        
        if (query.empty()) continue;
        
        std::vector<SearchResult> results = engine.search(query);
        
        if (results.empty()) {
            std::cout << "No results found." << std::endl;
        } else {
            std::cout << "Found " << results.size() << " match(es):" << std::endl;
            for (size_t i = 0; i < results.size() && i < 5; ++i) { // Show top 5
                std::cout << (i+1) << ". " << results[i].documentName 
                          << " (Score: " << results[i].score << ")" << std::endl;
            }
        }
    }

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
