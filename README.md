# Mini Search Engine (C++)

A high-performance text search engine implemented in C++ using advanced data structures.

## Features
*   **Inverted Index & Trie**: Efficiently maps words to documents.
*   **TF-IDF Ranking**: Ranks search results by relevance.
*   **Stopword Removal**: Filters out common words (e.g., "the", "is") for better accuracy.
*   **Multi-word Support**: Handles queries with multiple terms.
*   **Modular Design**: Clean separation of concerns (Indexer, Tokenizer, Storage).

## Data Data Structures
1.  **Trie**: Stores the vocabulary. Each leaf node allows O(L) storage and retrieval of internal `WordID`s.
2.  **Inverted Index**: Maps `WordID` -> `List of {DocID, Frequency}`.
3.  **Hash Map (DocumentStore)**: Maps `DocID` -> `Metadata` (File Path, Total Word Count).

## How to Build

### Prerequisites
*   C++17 compliant compiler (GCC, Clang, MSVC)
*   CMake (3.10+)

### Build Steps
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## How to Run
Ensure the `data` directory is in the same folder as the executable or in the parent directory.

```bash
./MiniSearchEngine
# On Windows
MiniSearchEngine.exe
```

## Usage
1.  Place text files to be indexed in the `data/` folder.
2.  Run the application.
3.  Enter queries when prompted.
    *   Example: `search engine`
    *   Example: `fast performance`
4.  Type `exit` to quit.

## Project Structure
*   `src/`: Source code (.cpp)
*   `include/`: Header files (.h)
*   `data/`: Sample datasets and stopwords
