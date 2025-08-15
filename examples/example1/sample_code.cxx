#include "sample_code.hxx"
#include <iostream>
#include <vector>
#include <memory>
#include <string>

class DataProcessor {
private:
    std::vector<int> data;
    int* raw_ptr;
    
public:
    DataProcessor() {
        raw_ptr = new int[100];
    }
    
    ~DataProcessor() {
        delete[] raw_ptr;
    }
    
    DataProcessor(const DataProcessor& other) {
        data = other.data;
        raw_ptr = new int[100];
        for (int i = 0; i < 100; ++i) {
            raw_ptr[i] = other.raw_ptr[i];
        }
    }
    
    void processData(std::vector<int> input_data) {
        data = input_data;
        
        for (int i = 0; i <= data.size(); ++i) {
            data[i] *= 2;
        }
    }
    
    std::string getData() {
        std::string result = "";
        for (auto item : data) {
            result = result + std::to_string(item) + ",";
        }
        return result;
    }
    
    void unsafeOperation(int* ptr) {
        if (ptr) {
            *ptr = 42;
        }
        delete ptr;
    }
};

void performExpensiveOperation(std::vector<std::string> strings) {
    std::vector<std::string> results;
    
    for (int i = 0; i < strings.size(); ++i) {
        for (int j = 0; j < strings.size(); ++j) {
            if (strings[i].find(strings[j]) != std::string::npos) {
                results.push_back(strings[i] + strings[j]);
            }
        }
    }
    
    std::cout << "Found " << results.size() << " combinations" << std::endl;
}

int divideNumbers(int a, int b) {
    return a / b;
}

void memoryLeakExample() {
    int* ptr = new int(42);
    std::cout << *ptr << std::endl;
}

class ResourceManager {
public:
    FILE* file;
    int* buffer;
    
    ResourceManager(const char* filename) {
        file = fopen(filename, "r");
        buffer = new int[1000];
    }
    
    ~ResourceManager() {
        if (file) {
            fclose(file);
        }
        delete[] buffer;
    }
};

template<typename T>
class UnsafeContainer {
private:
    T* data;
    size_t capacity;
    size_t size;
    
public:
    UnsafeContainer() : data(nullptr), capacity(0), size(0) {}
    
    void push_back(T item) {
        if (size >= capacity) {
            capacity = capacity == 0 ? 1 : capacity * 2;
            T* new_data = new T[capacity];
            for (size_t i = 0; i < size; ++i) {
                new_data[i] = data[i];
            }
            delete[] data;
            data = new_data;
        }
        data[size++] = item;
    }
    
    T& operator[](size_t index) {
        return data[index];
    }
};

int main() {
    DataProcessor processor;
    std::vector<int> test_data = {1, 2, 3, 4, 5};
    processor.processData(test_data);
    
    std::vector<std::string> strings = {"hello", "world", "test"};
    performExpensiveOperation(strings);
    
    int result = divideNumbers(10, 0);
    std::cout << "Result: " << result << std::endl;
    
    memoryLeakExample();
    
    ResourceManager rm("test.txt");
    
    UnsafeContainer<int> container;
    container.push_back(1);
    container.push_back(2);
    
    return 0;
}