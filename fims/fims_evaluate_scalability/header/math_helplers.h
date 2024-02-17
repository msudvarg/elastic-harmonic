#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>

// double interp2(Eigen::MatrixXd& xg, Eigen::MatrixXd& yg, Eigen::MatrixXd& Zp_s, double x, double y);
// double interp2(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::MatrixXd& z,
//                double xq, double yq);

Eigen::VectorXd linspace(double start, double end, int num_points);
Eigen::VectorXd logspace(double min, double max, int num_points);