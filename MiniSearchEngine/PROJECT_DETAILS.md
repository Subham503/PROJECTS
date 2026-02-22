# Upgraded Mini Search Engine - Technical Report

## 1. Project Overview
The **Upgraded Mini Search Engine** is a high-performance, file-based search engine written in **C++**. It is designed to index text files efficiently and retrieve relevant documents based on user queries using valid information retrieval techniques.

## 2. Technology Stack
*   **Language**: C++17 (Utilizing modern features like `std::unique_ptr`, `std::filesystem` concepts, structured bindings).
*   **Build System**: CMake (Industry standard for cross-platform C++ builds).
*   **Platform**: Cross-platform compatible (Windows via MinGW/MSVC, Linux, macOS).
*   **Standard Library**: Extensive use of STL (`vector`, `unordered_map`, `string`, `fstream`).

## 3. Core Data Structures
The engine leverages three primary data structures to ensure speed and modularity:

### A. Trie (Prefix Tree)
*   **Purpose**: Stores the **Vocabulary** (all unique words found in documents).
*   **Why used?**:
    *   **Space Efficiency**: Common prefixes are stored only once (e.g., "apple" and "apply" share "appl").
    *   **Speed**: Lookup time is **O(L)**, where L is the length of the word, independent of the number of words in the dictionary.
*   **Implementation**: Each `TrieNode` contains a map of children and a `wordID`. If a word ends at a node, it gets a unique integer ID.

### B. Inverted Index
*   **Purpose**: Maps words to the documents that contain them.
*   **Structure**: `HashMap<WordID, List<Posting>>`
    *   `Posting`: A simple struct containing `{DocID, Frequency}`.
*   **Why used?**: This is the standard structure for search engines. It allows instantaneous retrieval of all documents containing a specific word without scanning every file effectively **O(1)** lookup.

### C. Document Store (Hash Map)
*   **Purpose**: Stores metadata about documents.
*   **Structure**: `HashMap<DocID, DocumentMetadata>`
*   **Content**:
    *   `FilePath`: The location of the file.
    *   `TotalWordCount`: Required for calculating TF-IDF normalization.

## 4. Algorithms & Logic

### A. Indexing Pipeline
1.  **File Reading**: Reads content from `.txt` files.
2.  **Tokenization**:
    *   Splits text into words.
    *   **Normalization**: Converts to lowercase (Case-insensitive).
    *   **Filtering**: Removes non-alphanumeric characters.
3.  **Stopword Removal**: Filters out common noise words (e.g., "the", "and", "is") using a loaded list to improve relevance.
4.  **Trie Insertion**: Checks if word exists; if not, assigns a new `WordID`.
5.  **Index Update**: Appends the `DocID` and increments frequency in the `Inverted Index`.

### B. Ranking Algorithm (TF-IDF)
We use **Term Frequency-Inverse Document Frequency (TF-IDF)** to score results. This ensures that rare, significant words contribute more to the score than common words.

*   **TF (Term Frequency)**: How often word $t$ appears in doc $d$.
    $$TF = \frac{\text{count}(t, d)}{\text{total words in } d}$$
*   **IDF (Inverse Document Frequency)**: Significance of the word across all docs.
    $$IDF = \log(1 + \frac{N}{df_t})$$
    *   $N$: Total number of documents.
    *   $df_t$: Number of documents containing word $t$.
*   **Final Score**: Sum of $(TF \times IDF)$ for all query terms found in a document.

## 5. Folder Structure
*   `src/`: Contains implementation logic (`.cpp` files).
*   `include/`: Contains header definitions (`.h` files).
*   `data/`: Stores the text corpora and `stopwords.txt`.
*   `build/`: Temporary compilation artifacts.

## 6. How to Extend
*   **Boolean Queries**: Add parsing for `AND`, `OR`, `NOT` logic.
*   **Persistence**: Serialize the Trie and Index to disk to avoid re-indexing on every run.
*   **Stemming**: Implement Porter Stemmer to treat "running" and "run" as the same word.
