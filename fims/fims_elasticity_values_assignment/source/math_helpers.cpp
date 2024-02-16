#include "math_helplers.h"

#include <algorithm>
#include <iostream>
#include <limits>


Eigen::VectorXd linspace(double start, double end, int num_points) {
    Eigen::VectorXd result(num_points);
    double step = (end - start) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

Eigen::VectorXd logspace(double min, double max, int num_points) {
    double start = std::log10(min);
    double end = std::log10(max);
    Eigen::VectorXd base = linspace(start, end, num_points);
    for (int i = 0; i < num_points; ++i) {
        base[i] = std::pow(10, base[i]);
    }
    return base;
}


