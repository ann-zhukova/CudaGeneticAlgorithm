#pragma once
#include "Polynomial.h"

extern "C" {
Individual runGA_GPU(const Point* h_pts, int nPts, int popSize, int gens) ;
}