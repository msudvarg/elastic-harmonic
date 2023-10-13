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

    Task(float t_min, float t_max, float _c, float _e) :
        i {t_min, t_max},
        c {_c},
        e {_e} {}
};

using Tasks = std::vector<Task>;

void print_info(const Tasks & tasks);
