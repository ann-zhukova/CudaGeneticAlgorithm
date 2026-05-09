#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "GA_CPU.h"
#include "GA_GPU.cuh"

int main() {
    auto points = generateData(750);
    std::ofstream csv("research.csv");
    csv << "Device;PopSize;Gens;Time_ms;Fitness;a0;a1;a2;a3;a4\n";

    int pops[] = {1000, 2000, 5000};
    int gens[] = {50, 100, 200};

    for (int p : pops) {
        for (int g : gens) {
            // CPU
            auto s1 = std::chrono::high_resolution_clock::now();
            Individual b1 = GA_CPU::run(points, p, g);
            auto e1 = std::chrono::high_resolution_clock::now();
            auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(e1-s1).count();

            csv << "CPU;" << p << ";" << g << ";" << t1 << ";" << b1.fitness;
            for(int i=0; i<5; i++) csv << ";" << decode_cpu(b1.genome[i]);
            csv << "\n";

            // GPU
            auto s2 = std::chrono::high_resolution_clock::now();
            Individual b2 = runGA_GPU(points.data(), 750, p, g);
            auto e2 = std::chrono::high_resolution_clock::now();
            auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(e2-s2).count();

            csv << "GPU;" << p << ";" << g << ";" << t2 << ";" << b2.fitness;
            for(int i=0; i<5; i++) csv << ";" << decode_cpu(b2.genome[i]);
            csv << "\n";

            std::cout << "P:" << p << " G:" << g << " | CPU:" << t1 << "ms GPU:" << t2 << "ms\n";
        }
    }
    csv.close();
    return 0;
}