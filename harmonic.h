#pragma once

#include "task.h"

struct Harmonic {
    Interval i;
    int a;
};

struct Chain {
    std::vector<Harmonic> harmonics;
    float u_min, u_max, O_min, O_max;
    float A, B, Y;

};

struct Region {
    float lb, ub;
    bool flat;
    float A, B;
};

// using Harmonics = std::vector<int>;

struct Harmonic_Elastic {
    Tasks tasks;
    std::vector<Chain> chains;
    float u_min;

    Harmonic_Elastic(int n_tasks);

    void add_task(Task t);

    void generate();

    bool assign_periods_slow(float u_max);
};