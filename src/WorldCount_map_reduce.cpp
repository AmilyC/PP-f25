#include<iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <cstring>  
#include <cctype>
#include <map>
const char *stopwords[] = {"a", "about", 
    "above", "after", "again", "against", "ain",
    "all", "am", "an", "and", "any", "are", "aren", 
    "aren't", "as", "at", "be", "because", "been", 
    "before", "being", "below", "between", "both", 
    "but", "by", "can", "couldn", "couldn't", "d",
    "did", "didn", "didn't", "do", "does", "doesn", 
    "doesn't", "doing", "don", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn",
    "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
    "having", "he", "he'd", "he'll", "her", "here", "hers", "herself",
    "he's", "him", "himself", "his", "how", "i", "i'd", "if", "i'll",
    "i'm", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll",
    "it's", "its", "itself", "i've", "just", "ll", "m", "ma", "me", 
    "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my",
    "myself", "needn", "needn't","no", "nor", "not", "now", "o", "of", 
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", 
    "re", "s", "same", "shan", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn", "shouldn't", "should've", "so", 
    "some", "such", "t", "than", "that", "that'll", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", 
    "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", 
    "wasn't", "we", "we'd", "we'll", "we're", "were", "weren", "weren't", 
    "we've", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've",NULL};
const char *dataset_path[] = {
    "../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt"
};

bool is_stopword(char *word) {
    for (int i = 0; stopwords[i] != NULL; i++) {
        if (strcmp(stopwords[i], word) == 0) {
            return true; 
        }
    }
    return false; 
}
struct KVPair {
    char* key; // 
    int value;
};
struct ShuffledData {
    char* key;
    std::vector<int> values; //
};
//map
std::vector<KVPair> map(char* article_chunk) {
    std::vector<KVPair> mapped_data;
    const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";
    char* saveptr;
    
    char* token;
    token = strtok_r(article_chunk, delimiters, &saveptr); // )

    while (token != NULL) {
        // ... (tolower, strlen, is_stopword, isalnum) ...
        
        // turn to lower
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]);
        }

        // get the next token
        if (strlen(token) <= 2) {
            
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        // check if stop words
        if (is_stopword(token)) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        // fliter out the word that has number or special char eg:123, c--
        bool is_word_char = true;
        if (strlen(token) == 0) { // strtok_r somtime generate empty token
             is_word_char = false;
        }

        for (int i = 0; token[i]; i++) {
            // isalnum() check if alpha or num
            if (!isalnum(token[i]) && token[i] != '_') { 
                is_word_char = false;
                break;
            }
        }

        if (!is_word_char) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }
        // "(Emit)" (key, 1)
        KVPair pair;
        // strdup = (malloc + strcpy)，to build a new memory
        // 
        pair.key = strdup(token); 
        pair.value = 1;
        
        mapped_data.push_back(pair);

        token = strtok_r(NULL, delimiters, &saveptr);
    }
    return mapped_data;
}


//shuffle
std::map<std::string, std::vector<int>> shuffle(const std::vector<KVPair>& mapped_data) {
    
    // use std::map 
    //  sort by key
    std::map<std::string, std::vector<int>> shuffled_map;

    for (const auto& pair : mapped_data) {
        // make  (key, 1) insert to  key  vector 
        shuffled_map[std::string(pair.key)].push_back(pair.value);
        
        // release pair key;
        free(pair.key); 
    }
    return shuffled_map;
}

//reduce
KVPair reduce(const std::string& key, const std::vector<int>& values) {
    int sum = 0;
    for (int val : values) {
        sum += val;
    }

    KVPair result;
    result.key = strdup(key.c_str()); // 再次複製
    result.value = sum;
    
    return result;
}


int main(){

    // read file

    std::ifstream ifs(dataset_path[0], std::ios::in);
    if (!ifs.is_open()) {
        std::cout << "Failed to open file.\n";
        return 1;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string str(ss.str());
    
    ifs.close(); 

    
    std::unordered_map<std::string, int> word_counts;



    // split it into words
    char* token;
    const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";

    char* article_buffer = &str[0];
    std::vector<KVPair> mapped_results = map(article_buffer);

    // 2. Shuffle
    std::map<std::string, std::vector<int>> shuffled_results = shuffle(mapped_results);
    
    // 3. Reduce
    std::vector<KVPair> final_results;
    for (const auto& group : shuffled_results) {
        final_results.push_back(reduce(group.first, group.second));
    }

    // 4. print
    std::cout << "--- Word Count Results ---" << std::endl;
    for (auto& pair : final_results) {
        if (pair.value > 100) {
            std::cout << pair.key << ": " << pair.value << std::endl;
        }
        free(pair.key); // release reduce  key
    }

    return 0;
}