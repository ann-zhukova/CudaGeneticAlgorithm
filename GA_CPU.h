#pragma once
#include "Polynomial.h"
#include <algorithm>

class GA_CPU {
public:
    static Individual run(const std::vector<Point>& points, int popSize, int generations) {
        std::vector<Individual> population(popSize);

        for (auto& ind : population) {
            for (int i = 0; i < 5; ++i) {
                float val = (rand() % 200 - 100) / 20.0f;
                std::memcpy(&ind.genome[i], &val, 4);
            }
        }

        for (int gen = 0; gen < generations; ++gen) {
            for (auto& ind : population) {
                float maxErr = 0.0f;
                float c[5];
                bool bad = false;
                for(int k=0; k<5; k++) {
                    c[k] = decode_cpu(ind.genome[k]);
                    if (!std::isfinite(c[k])) bad = true;
                }

                if (bad) { ind.fitness = 1e10f; continue; }

                for (const auto& p : points) {
                    float f = c[0] + c[1]*p.x + c[2]*p.x*p.x + c[3]*p.x*p.x*p.x + c[4]*p.x*p.x*p.x*p.x;
                    float err = std::abs(f - p.y);
                    if (err > maxErr) maxErr = err;
                }
                ind.fitness = std::isfinite(maxErr) ? maxErr : 1e10f;
            }

            std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });

            for (int i = popSize / 2; i < popSize; ++i) {
                int p1 = rand() % (popSize / 2);
                int p2 = rand() % (popSize / 2);
                int cp = rand() % 159 + 1;
                for (int j = 0; j < 5; j++) {
                    int start = j * 32;
                    if (start + 32 <= cp) population[i].genome[j] = population[p1].genome[j];
                    else if (start >= cp) population[i].genome[j] = population[p2].genome[j];
                    else {
                        unsigned int mask = (1U << (cp - start)) - 1;
                        population[i].genome[j] = (population[p1].genome[j] & mask) | (population[p2].genome[j] & ~mask);
                    }
                }
                if ((rand() % 100) < 10) population[i].genome[rand()%5] ^= (1U << (rand()%32));
            }
        }
        return population[0];
    }
};