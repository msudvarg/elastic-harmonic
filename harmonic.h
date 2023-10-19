#pragma once

#include "task.h"

struct Harmonic {
    Interval i;
    int a;
};

struct Harmonic_Projection {
    Interval i;
    std::vector<int> multiples;
};

struct Chain {
    std::vector<Harmonic> harmonics;
    float u_min, u_max, O_min, O_max;
    float A, B, Y;

};

struct Region {
    float lb, ub;
    struct Chain * chain;
    friend bool operator < (const Region & a, const Region & b);
};

// using Harmonics = std::vector<int>;

struct Harmonic_Elastic {

    Tasks tasks;
    std::vector<Chain> chains;
    std::vector<Region> regions;
    float u_min;

    Harmonic_Elastic(int n_tasks);

    void add_task(Task t);

    void generate();
    void generate_intersections();

    bool assign_periods_slow(float u_max);
};