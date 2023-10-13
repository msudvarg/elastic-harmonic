#pragma once
#include <vector>

struct Interval {
    float t_min, t_max;
};

struct Task {
    Interval i;
    float t, c, e;

    inline float u_min() const {
        return c/(float)i.t_max;
    }

    inline float u_max() const {
        return c/(float)i.t_min;
    }

    inline float u() const {
        return c/(float)t;
    }
};

using Tasks = std::vector<Task>;

void print_info(const Tasks & tasks);
