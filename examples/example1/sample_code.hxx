#ifndef SAMPLE_CODE_HXX
#define SAMPLE_CODE_HXX

#include <vector>
#include <string>
#include <memory>

class DataProcessor {
private:
    std::vector<int> data;
    int* raw_ptr;
    
public:
    DataProcessor();
    ~DataProcessor();
    DataProcessor(const DataProcessor& other);
    
    void processData(std::vector<int> input_data);
    std::string getData();
    void unsafeOperation(int* ptr);
};

void performExpensiveOperation(std::vector<std::string> strings);

int divideNumbers(int a, int b);

void memoryLeakExample();

class ResourceManager {
public:
    FILE* file;
    int* buffer;
    
    ResourceManager(const char* filename);
    ~ResourceManager();
};

template<typename T>
class UnsafeContainer {
private:
    T* data;
    size_t capacity;
    size_t size;
    
public:
    UnsafeContainer();
    
    void push_back(T item);
    T& operator[](size_t index);
};

template<class T>
inline void problematic_function(T value) {
    static T cached_value;
    cached_value = value;
}

#define UNSAFE_MACRO(x) ((x) > 0 ? (x) : -(x))

typedef struct {
    char name[50];
    int age;
    float* scores;
} Person;

extern int global_counter;

namespace utils {
    inline std::string concatenate(std::string a, std::string b) {
        return a + b;
    }
    
    template<typename T>
    T max(T a, T b) {
        return a > b ? a : b;
    }
}

class SingletonExample {
private:
    static SingletonExample* instance;
    SingletonExample() {}
    
public:
    static SingletonExample* getInstance() {
        if (!instance) {
            instance = new SingletonExample();
        }
        return instance;
    }
};

#endif // SAMPLE_CODE_HXX