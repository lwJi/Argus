#include <iostream>
#include <vector>
#include <memory>

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
};

class Rectangle : public Shape {
private:
    double width_, height_;

public:
    Rectangle(double width, double height) 
        : width_(width), height_(height) {}
    
    double area() const override {
        return width_ * height_;
    }
};

class Circle : public Shape {
private:
    double radius_;

public:
    explicit Circle(double radius) : radius_(radius) {}
    
    double area() const override {
        return 3.14159 * radius_ * radius_;
    }
};

void process_shapes(const std::vector<std::unique_ptr<Shape>>& shapes) {
    for (const auto& shape : shapes) {
        std::cout << "Area: " << shape->area() << std::endl;
    }
}