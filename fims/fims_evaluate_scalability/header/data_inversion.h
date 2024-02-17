#pragma once
#include "read_house_keeping.h"

#include <string>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <atomic>


class Data_Inversion {
public:
    Data_Inversion(std::mutex& data_mutex, std::condition_variable& data_cv, std::deque<House_Keeping>& hk_deque,
               std::deque<std::pair<double, std::vector<cv::Point2d>>> & particle_deque, std::atomic<double>& frame_time);
    
    // ~Data_Inversion();

    void start_inversion();
    void stop_inversion();

private:

    void perform_data_inversion();
    void perform_data_inversion2();
    double current_HV_level, current_T, current_P, current_Qa;
    double current_time;
    double Zp_s_max;
    double Zp_s_min;
    House_Keeping pre_hk;

    // const double mfp_1atm_298K = 6.62e-8;
    // const double e = 1.602176634e-19;
    Eigen::VectorXd xg;
    Eigen::VectorXd yg;
    Eigen::MatrixXd Zp_s;
    std::vector<Eigen::VectorXd> xg_cases;
    std::vector<Eigen::VectorXd> yg_cases;
    std::vector<Eigen::MatrixXd> Zp_s_cases;
    // Eigen::VectorXd Dp_bnds;
    // Eigen::VectorXd Zp_bnds;
    Eigen::MatrixXd Gamma;


    std::thread data_inversion_thread_;
    std::mutex& data_mutex_;
    std::condition_variable& data_cv_;
    std::deque<House_Keeping>& hk_deque_; 
    std::deque<std::pair<double, std::vector<cv::Point2d>>>& particle_deque_;
    bool stop_thread_;
    std::atomic<double>& frame_time_;

    House_Keeping interp_hk(House_Keeping& hk_pre, House_Keeping& hk_cur, double time);
    int find_HV_level1(double Vm, const std::vector<int>& Vm_range);
    int find_HV_level2(double Vm, const std::vector<int>& Vm_range);
    std::string compose_path(int case_number, std::string type);
    // Eigen::MatrixXd cvMatToEigen(const cv::Mat& cvMat);
    // std::pair<Eigen::VectorXd, Eigen::VectorXd> histcounts(std::vector<cv::Point2d>& particles, const Eigen::VectorXd& bnds);
    Eigen::VectorXd histcounts(std::vector<cv::Point2d>& particles, const Eigen::VectorXd& Zp_bnds);
    std::vector<double> cal_Zp_s(std::vector<cv::Point2d>& particles);
    double interp2(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::MatrixXd& z,
               double xq, double yq);
    Eigen::VectorXd interp1(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& xi);
    Eigen::VectorXd smoothVector(const Eigen::VectorXd& n, double b);
    Eigen::VectorXd Twomey_inv(const Eigen::MatrixXd& Gamma, const Eigen::VectorXd& R);
    Eigen::VectorXd linear_inv(const Eigen::MatrixXd& Gamma, const Eigen::VectorXd& R);
    Eigen::VectorXd interpolate_n2(Eigen::VectorXd& n0, Eigen::VectorXd& Dpmin, Eigen::VectorXd& Dp, 
                                    Eigen::VectorXd& Dpmax, Eigen::MatrixXd& Dp_fixed_set, char mode = 'N');
    

};




