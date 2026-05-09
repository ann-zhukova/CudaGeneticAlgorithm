#pragma once
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>

struct Point {
    float x, y;
};

struct Individual {
    unsigned int genome[5];
    float fitness;
};

inline float decode_cpu(unsigned int bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline std::vector<Point> generateData(int n) {
    std::vector<Point> points;
    srand(42);
    for (int i = 0; i < n; ++i) {
        float x = (float)i / n * 10.0f;
        float y = 1.2f * x * x - 2.5f * x + 3.0f + ((rand() % 100) / 500.0f);
        points.push_back({ x, y });
    }
    return points;
}