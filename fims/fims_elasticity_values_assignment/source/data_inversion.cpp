#include "data_inversion.h"
#include "paramt_FIMS.h"
#include "math_helplers.h"
#include "compute_inversion_matrix.h"

#include <cmath>
#include <thread>
#include <chrono>
#include <unsupported/Eigen/NNLS>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <sys/resource.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <cstdlib>
#include <ctime>

Data_Inversion::Data_Inversion(std::mutex& data_mutex, std::condition_variable& data_cv, std::deque<House_Keeping>& hk_deque, 
            std::deque<std::pair<double, std::vector<cv::Point2d>>>& particle_deque, std::atomic<double>& frame_time)
    : data_mutex_(data_mutex), data_cv_(data_cv), hk_deque_(hk_deque), particle_deque_(particle_deque), frame_time_(frame_time), stop_thread_(false) {
        pre_hk = House_Keeping(-1.0);
    }


void Data_Inversion::start_inversion() {
    data_inversion_thread_ = std::thread(&Data_Inversion::perform_data_inversion, this);
    // // Pin the thread to core 2.
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(2, &cpuset);
    // pthread_t native_thread = data_inversion_thread_.native_handle();

    // if (pthread_setaffinity_np(native_thread, sizeof(cpu_set_t), &cpuset)) {
    //     std::cerr << "Could not set thread to CPU 2" << std::endl;
    // }
    // struct sched_param schedParam;
    // schedParam.sched_priority = 97;  // Set the priority to 97
    // if (pthread_setschedparam(native_thread, SCHED_FIFO, &schedParam) != 0) {
    //     std::cerr << "Failed to set thread scheduling policy and priority" << std::endl;
    // }
}


void Data_Inversion::stop_inversion() {
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        stop_thread_ = true;
    }
    data_cv_.notify_all();
    if (data_inversion_thread_.joinable()) {
        data_inversion_thread_.join();
    }
}

void Data_Inversion::perform_data_inversion() {
    paramt_FIMS& fims = paramt_FIMS::getInstance();
    Compute_Inversion_Matrix& inversion_matrix = Compute_Inversion_Matrix::getInstance();
    // std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    for(int i = 0; i < fims.case_num.size(); ++i) {
        std::string xg_path = compose_path(fims.case_num[i], "xg");
        Eigen::VectorXd cur_xg = inversion_matrix.load_vector_from_csv(xg_path);
        xg_cases.push_back(cur_xg);

        std::string yg_path = compose_path(fims.case_num[i], "yg");
        Eigen::VectorXd cur_yg = inversion_matrix.load_vector_from_csv(yg_path);
        yg_cases.push_back(cur_yg);

        std::string Zp_s_path = compose_path(fims.case_num[i], "Zp_s");
        Eigen::MatrixXd cur_Zp_s = inversion_matrix.load_matrix_from_csv(Zp_s_path);
        Zp_s_cases.push_back(cur_Zp_s);
    }


    std::ofstream fixed_result_cpp("../result_/n_Dp_fixed.txt");
    // std::ofstream result_cpp("../../generate_result/result_/n_cpp.txt");
    int loop = 0;
    auto inversion_duration = std::chrono::milliseconds(int(fims.data_inversion_duration));
    // std::ofstream inversion_time("../result_/run_time/inversion_time.txt");
    // std::ofstream loop_time("../result_/run_time/loop_time.txt");
    // std::ofstream inversion_time_user("../result_/run_time/inversion_time_user.txt");
    // std::ofstream inversion_time_sys("../result_/run_time/inversion_time_sys.txt");

    // std::vector<double> inversion_time_(12000, 0);

    // struct rusage usage_start, usage_end;
    // getrusage(RUSAGE_THREAD, &usage_start);
    // struct timespec cpustart, cpuend; 
    // clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpustart);
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        std::cerr << "Processing:" << loop++ << "th time bin" << std::endl;
        std::unique_lock<std::mutex> lock(data_mutex_);
        data_cv_.wait(lock, [&] {
            if (stop_thread_) return true;
            if (hk_deque_.empty() || particle_deque_.empty()) {
                return false;
            }
            if (hk_deque_.front().UTC < particle_deque_.front().first + fims.bnd_interval / double(2*3600)) {
                if (hk_deque_.front().UTC > pre_hk.UTC) {
                    pre_hk = hk_deque_.front();
                }
                hk_deque_.pop_front();
                return false;
            }
            if (frame_time_ - particle_deque_.front().first < double(fims.bnd_interval + 2.5) / double(3600.0)) {
                return false;
            }
            return true;
        });

        if (stop_thread_) {
            // std::ofstream inversion_time("../result_/run_time/inversion_time.txt");
            // for(int i = 0; i < inversion_time_.size() - 1; ++i){
            //     if(inversion_time_[i] != 0 && inversion_time_[i + 1] != 0)
            //     inversion_time << inversion_time_[i] << " " << std::flush;
            // }
            // loop_time.close();
            // inversion_time_user.close();
            // inversion_time_sys.close();
            // inversion_time.close();
            break;
        }    

        auto cur_hk = hk_deque_.front();
        std::vector<cv::Point2d> cur_particles = particle_deque_.front().second;
        double cur_time = particle_deque_.front().first;
        particle_deque_.pop_front();
        lock.unlock();
        data_cv_.notify_all();
        
        House_Keeping cur_housekeeping = House_Keeping();
        if(pre_hk.UTC == -1) {
            pre_hk = cur_hk;
            cur_housekeeping = cur_hk;
        } 
        else {
            cur_housekeeping = interp_hk(pre_hk, cur_hk, cur_time + fims.bnd_interval/(2*3600));
        }
     
        if(cur_particles.empty()) {
            // result_cpp <<  Eigen::RowVectorXd::Zero(30) << std::endl;
            Eigen::VectorXd ret = Eigen::RowVectorXd::Zero(30);
            // result_cpp << ret << std::endl;
            // result_cpp <<  "NaN" << std::endl << std::flush;
            fixed_result_cpp << ret << std::endl;
            fixed_result_cpp << "NaN" << std::endl << std::flush;
            // fifo << cur_time << " " << ret << "NaN" << std::flush;
            continue;
        }

        int HV_level = fims.HV_level_method == 1 ? find_HV_level1(cur_housekeeping.HV, fims.Vm_range) 
                    : find_HV_level2(-cur_housekeeping.HV_set, fims.Vm_range);
        
        if(HV_level == -1) {
            std::cerr << "WARNING: HV_level not found." << std::endl;
            continue;
        }

        if(cur_housekeeping.Qa == -1 || cur_housekeeping.Qa < fims.Qa_min_for_inv) {
            std::cerr << "WARNING: Aerosol flow rate too low for inversion." << std::endl;
            continue;
        }

        Eigen::VectorXd Dp_bnds;
        Eigen::VectorXd Zp_bnds;

        if(HV_level != current_HV_level || (std::abs((cur_housekeeping.Pabs - current_P) / current_P) > 0.02) ||
                (std::abs(cur_housekeeping.Ta - current_T) > 1)) 
        {
            current_HV_level=HV_level;
            current_T = cur_housekeeping.Ta;
            current_P = cur_housekeeping.Pabs;
            current_Qa = cur_housekeeping.Qa;
            
            xg = xg_cases[HV_level];
            yg = yg_cases[HV_level];
            Zp_s = Zp_s_cases[HV_level];

            
            Zp_s_max = interp2(xg, yg, Zp_s, fims.x_max, fims.y_min);
            Zp_s_min = interp2(xg, yg, Zp_s, fims.x_min, fims.y_max);

            Dp_bnds = logspace(inversion_matrix.Zp_to_Dp(Zp_s_max, current_T+273.15, current_P, 1), inversion_matrix.Zp_to_Dp(Zp_s_min, current_T+273.15, current_P, 1), fims.N_Dp_bin + 1);
            Zp_bnds = inversion_matrix.Dps_to_Zps(Dp_bnds, current_T+273.15, current_P,1);
            
            double factor = current_Qa * 1000 / 60 * fims.dt;

            Gamma = inversion_matrix.FIMS_inversion_matrix(factor, Zp_bnds, Dp_bnds, current_T+273.15, current_P,
                                        fims.chgs, fims.Veff[HV_level], fims.charger_type, fims.inlet_ver, fims.case_num[HV_level]);
        } 
        else {
            Dp_bnds = logspace(inversion_matrix.Zp_to_Dp(Zp_s_max, current_T+273.15, current_P, 1), inversion_matrix.Zp_to_Dp(Zp_s_min, current_T+273.15, current_P, 1), fims.N_Dp_bin+1);
            Zp_bnds = inversion_matrix.Dps_to_Zps(Dp_bnds, current_T+273.15, current_P,1);
            Gamma = cur_housekeeping.Qa / current_Qa * Gamma;        
            current_Qa = cur_housekeeping.Qa;
        }

        std::vector<double> tem_Zp = cal_Zp_s(cur_particles);

        Eigen::VectorXd R = histcounts (cur_particles, Zp_bnds.reverse());
        R.reverseInPlace();

        Eigen::VectorXd invert_data = Twomey_inv(Gamma, R);
        Eigen::VectorXd n = invert_data.transpose() * cur_housekeeping.Pamb / cur_housekeeping.Pabs;
        // Initialize matrices and vectors
        Eigen::VectorXd Dpmin, Dpmax, Dp, n_Dp_fixed;
        // Set Dpmin and Dpmax using segment of Dp_bnds
        Dpmin = Dp_bnds.head(Dp_bnds.size() - 1);
        Dpmax = Dp_bnds.tail(Dp_bnds.size() - 1);
        // Calculate Dp
        Dp = (Dpmin.array() * Dpmax.array()).sqrt();
        // Calculate N
        double N = (n.array() * (Dpmax.array() / Dpmin.array()).log()).sum();

        n_Dp_fixed = interpolate_n2(n, Dpmin, Dp, Dpmax, fims.Dp_fixed_set);
        // result_cpp << n << std::endl;
        // result_cpp << "NaN" << std::endl << std::flush;
        fixed_result_cpp << n_Dp_fixed << std::endl;
        fixed_result_cpp << "NaN" << std::endl << std::flush;

        // fifo << cur_time << " " << n_Dp_fixed << "NaN" << std::flush;
        // if (!fifo.good()) {
        //     std::cerr << "Failed to write to FIFO" << std::endl;
        // }
        // fifo_fixed << n_Dp_fixed << "NaN" << std::flush;

        // getrusage(RUSAGE_THREAD, &usage_end);  
        // double user_time_millis = ((usage_end.ru_utime.tv_sec - usage_start.ru_utime.tv_sec) * 1000.0) + 
        //                   (usage_end.ru_utime.tv_usec - usage_start.ru_utime.tv_usec) / 1000.0;
        // double system_time_millis = ((usage_end.ru_stime.tv_sec - usage_start.ru_stime.tv_sec) * 1000.0) + 
        //                     (usage_end.ru_stime.tv_usec - usage_start.ru_stime.tv_usec) / 1000.0;
        // usage_start = usage_end;
        // inversion_time_user << user_time_millis << " " << std::flush;
        // inversion_time_sys << system_time_millis << " " << std::flush;
        // clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpuend);
        // double elapsed = (cpuend.tv_sec - cpustart.tv_sec) * 1000 + (cpuend.tv_nsec - cpustart.tv_nsec) / 1e6;
        // auto stop = std::chrono::high_resolution_clock::now(); 
        // auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        // inversion_time << total_duration.count() << " " << std::flush;
        // inversion_time_[loop++] = total_duration.count();
        // start += inversion_duration;
        // std::this_thread::sleep_until(start);
    }
}


House_Keeping Data_Inversion::interp_hk(House_Keeping& hk_pre, House_Keeping& hk_cur, double time) {
    House_Keeping hk = hk_pre;
    if(hk_pre.UTC == hk_cur.UTC)
        return hk;
    hk.UTC = time;
    double t = (time -hk_pre.UTC ) / (hk_cur.UTC - hk_pre.UTC);
    hk.HV_set = hk_pre.HV_set + t * (hk_cur.HV_set - hk_pre.HV_set);
    hk.Qa_set = hk_pre.Qa_set + t * (hk_cur.Qa_set - hk_pre.Qa_set);
    hk.Ta = hk_pre.Ta + t * (hk_cur.Ta - hk_pre.Ta);
    hk.Pabs = hk_pre.Pabs + t * (hk_cur.Pabs - hk_pre.Pabs);
    hk.Pamb = hk_pre.Pamb + t * (hk_cur.Pamb - hk_pre.Pamb);
    hk.Qa = hk_pre.Qa + t * (hk_cur.Qa - hk_pre.Qa);
    hk.Qhs = hk_pre.Qhs + t * (hk_cur.Qhs - hk_pre.Qhs);
    hk.RHa = hk_pre.RHa + t * (hk_cur.RHa - hk_pre.RHa);
    return hk;
}

int Data_Inversion::find_HV_level1(double Vm, const std::vector<int>& Vm_range) {
    for (int i = 0; i < Vm_range.size() - 1; ++i) {
        if (Vm_range[i] <= Vm && Vm_range[i + 1] >= Vm) {
            return i;
        }
    }
    return -1;
}

int Data_Inversion::find_HV_level2(double Vm, const std::vector<int>& Vm_range) {
    for (int i = 0; i < Vm_range.size(); ++i) {
        if (std::abs((Vm - Vm_range[i]) / Vm_range[i]) < 1e-6) {
            return i;
        }
    }
    return -1;
}

std::string Data_Inversion::compose_path(int case_number, std::string type) {
    std::stringstream path_ss;
    if(type == "xg" || type == "yg" || type == "Zp_s") {
        path_ss << "../transfer_function/case_" << case_number << "/position2Zp_s/" << type << ".csv";
    }
    else {
        std::runtime_error("Error: path does not exist!");
    }
    std::string path = path_ss.str();        
    return path;
}


std::vector<double> Data_Inversion::cal_Zp_s(std::vector<cv::Point2d>& particles) {
    std::vector<double> temp_Zp;
    for (cv::Point2d& point : particles) {
        double val = interp2(xg, yg, Zp_s, point.x, point.y);
        temp_Zp.push_back(val);
    }
    return temp_Zp;
}

Eigen::VectorXd Data_Inversion::histcounts(std::vector<cv::Point2d>& particles, const Eigen::VectorXd& Zp_bnds) {
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(30);
    for (cv::Point2d& point : particles) {
        double val = interp2(xg, yg, Zp_s, point.x, point.y);
        if(std::isnan(val))
            continue;
        auto it = std::upper_bound(Zp_bnds.data(), Zp_bnds.data() + Zp_bnds.size(), val);
        int index = std::distance(Zp_bnds.data(), it);
        if (index > 0 && index < Zp_bnds.size())
            counts[ index - 1]++;
    }
    return counts;
}


double Data_Inversion::interp2(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::MatrixXd& z,
               double xq, double yq) {

    int ix = -1;
    int iy = -1;

    if (z.rows() != y.size() && z.cols() != x.size()) {
        throw std::runtime_error("Dimensions of z do not match sizes of x and y.");
    }

    auto x_upper = std::upper_bound(x.data(), x.data() + x.size(), xq);
    ix = x_upper - x.data() - 1;
    if (ix < 0 || ix >= x.size() - 1) {
        // std::cerr << "Warning: xq is outside the grid." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    auto y_upper = std::upper_bound(y.data(), y.data() + y.size(), yq);
    iy = y_upper - y.data() - 1;
    if (iy < 0 || iy >= y.size() - 1) {
        // std::cerr << "Warning: yq is outside the grid." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    double tx = (xq - x(ix)) / (x(ix + 1) - x(ix));
    double ty = (yq - y(iy)) / (y(iy + 1) - y(iy));


    double interpolated_value = (1 - tx) * (1 - ty) * z(iy, ix) +
                                tx * (1 - ty) * z(iy, ix + 1) +
                                (1 - tx) * ty * z(iy + 1, ix) +
                                tx * ty * z(iy + 1, ix + 1);

    return interpolated_value;
}


Eigen::VectorXd Data_Inversion::smoothVector(const Eigen::VectorXd& n, double b) {
    Eigen::VectorXd n1 = n;
    n1(0) = (1 - b) * n(0) + b * n(1);
    n1(n1.size() - 1) = (1 - b) * n(n.size() - 1) + b * n(n.size() - 2);
    n1.segment(1, n1.size() - 2) = b * n.head(n.size() - 2) + (1 - 2 * b) * n.segment(1, n.size() - 2) + b * n.tail(n.size() - 2);
    return n1;
}

Eigen::VectorXd Data_Inversion::Twomey_inv(const Eigen::MatrixXd& Gamma, const Eigen::VectorXd& R) {
    int max_iter = 1000;

    if (R.isZero()) {
        return Eigen::VectorXd::Zero(Gamma.cols());
    }

    Eigen::VectorXd R_nonzero = R;
    Eigen::VectorXd error = R_nonzero.array().sqrt();
    error = (error.array() == 0).select(0.001, error);

    Eigen::NNLS<Eigen::MatrixXd> nnls(Gamma);
    nnls.setMaxIterations(100);
    nnls.setTolerance(1e-9);
    Eigen::VectorXd n = nnls.solve(R);

    n = smoothVector(n, 1.0/3.0);

    double chi_sqrd = 1000;
    double chi_sqrd_reduction = 1;
    int iter = 0;

    while (abs(chi_sqrd_reduction) > 0.001 && chi_sqrd > 1 && iter<=max_iter ) {

        Eigen::VectorXd X = R_nonzero.array() / (Gamma * n).array();

        for (int i = 0; i < X.size(); ++i) {
            if (!std::isnan(X(i)) && !std::isinf(X(i))) {
                n = n.array() * ((1.0 + (X(i) - 1.0) * Gamma.row(i).array()).transpose());
            }
        }

        Eigen::VectorXd R_new = Gamma * n;
        double chi_sqrd_new = ((R_new - R_nonzero).array() / error.array()).pow(2).sum() / R_nonzero.size();
        chi_sqrd_reduction = (chi_sqrd - chi_sqrd_new) / chi_sqrd_new;

        chi_sqrd = chi_sqrd_new;
        iter++;
        
    }

    if (iter == max_iter + 1) {
        // std::cerr << "max iteration reached" << std::endl;
    }

    return n;
}

Eigen::VectorXd Data_Inversion::linear_inv(const Eigen::MatrixXd& Gamma, const Eigen::VectorXd& R) {
    Eigen::MatrixXd Gamma_square = Gamma.transpose() * Gamma;
    Eigen::MatrixXd R_new = Gamma.transpose() * R;
    Eigen::VectorXd n = Gamma_square.inverse() * R_new;

    return n;
}

Eigen::VectorXd Data_Inversion::interpolate_n2(Eigen::VectorXd& n0, Eigen::VectorXd& Dpmin, Eigen::VectorXd& Dp, Eigen::VectorXd& Dpmax, Eigen::MatrixXd& Dp_fixed_set, char mode){
    if(Dp.size() != n0.size())
        std::cerr << "The dimensions of n0 and Dp0 do not match" << std::endl;
    if(Dp_fixed_set.rows() != 3)
        std::cerr << "The number of rows in Dp do not match" << std::endl;

    Eigen::RowVectorXd Dp_bnd(Dp_fixed_set.cols() + 1);
    Dp_bnd << Dp_fixed_set.row(0), Dp_fixed_set(Dp_fixed_set.rows()-1, Dp_fixed_set.cols()-1);
    
    Eigen::VectorXd n = Eigen::VectorXd::Constant(Dp_fixed_set.cols(), std::numeric_limits<double>::quiet_NaN());

    std::vector<int> valid_indices;
    for(int i = 0; i < n0.size(); i++) {
        if(!std::isnan(n0(i))) {
            valid_indices.push_back(i);
        }
    }

    if(!valid_indices.empty()) {
        Eigen::VectorXd new_n0(valid_indices.size());
        Eigen::VectorXd new_Dpmin(valid_indices.size());
        Eigen::VectorXd new_Dp(valid_indices.size());
        Eigen::VectorXd new_Dpmax(valid_indices.size());
        for(int i = 0; i < valid_indices.size(); i++) {
            new_n0(i) = n0(valid_indices[i]);
            new_Dpmin(i) = Dpmin(valid_indices[i]);
            new_Dp(i) = Dp(valid_indices[i]);
            new_Dpmax(i) = Dpmax(valid_indices[i]);
        }

        Eigen::VectorXd Dp0_bnd(new_Dpmin.rows() + 1);
        Dp0_bnd << new_Dpmin, new_Dpmax(new_Dpmax.size()-1);

        if(mode == 'N') {
            Eigen::VectorXd N0 = new_n0.array() * (new_Dpmax.array().log() - new_Dpmin.array().log());

            int start_index = (Dp_fixed_set.row(0).array() < new_Dpmin(0)).count();
            int end_index = (Dp_fixed_set.row(2).array() <= new_Dpmax(new_Dpmax.size() - 1)).count() - 1;

            if (start_index > end_index)
                return Eigen::VectorXd::Zero(30);

            Eigen::ArrayXd cumsum_N0(N0.size() + 1);
            cumsum_N0(0) = 0; 
            for (int i = 1; i < cumsum_N0.size(); ++i) {
                cumsum_N0(i) = cumsum_N0(i - 1) + N0(i - 1); 
            }

            Eigen::VectorXd log_Dp0_bnd = Dp0_bnd.array().log();
            Eigen::VectorXd log_Dp_bnd = Dp_bnd.segment(start_index, end_index - start_index + 2).array().log();
            Eigen::VectorXd N = interp1(log_Dp0_bnd, cumsum_N0, log_Dp_bnd);
            Eigen::VectorXd N_ = N.tail(N.size()-1) - N.head(N.size()-1);
            
            Eigen::VectorXd k = (Dp_fixed_set.row(2).segment(start_index, end_index - start_index + 1).array() /
            Dp_fixed_set.row(0).segment(start_index, end_index - start_index + 1).array()).log();
            n.segment(start_index, end_index - start_index + 1) = N_.array() / k.array();
            

        }
        else if(mode == 'n') {
            int start_index = (Dp_fixed_set.row(1).array() < new_Dp(0)).count();
            int end_index = (Dp_fixed_set.row(1).array() <= Dp(Dp.size() - 1)).count() - 1;
            if (start_index > end_index)
                return Eigen::VectorXd::Zero(30);

            Eigen::VectorXd x = new_Dp;
            Eigen::VectorXd xi = Dp_fixed_set.row(1).segment(start_index, end_index - start_index + 1);
            Eigen::VectorXd N = interp1(x, n0, xi);

            n.segment(start_index, end_index - start_index + 1) = N;
        }
    }
    return n;
}

Eigen::VectorXd Data_Inversion::interp1(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::VectorXd& xi) {
    Eigen::VectorXd yi(xi.size());
    
    for(int i=0; i < xi.size(); ++i) {
        if(xi[i] <= x[0]) {
            yi[i] = std::numeric_limits<double>::quiet_NaN();
        }
        else if(xi[i] >= x[x.size() - 1]) {
            yi[i] = std::numeric_limits<double>::quiet_NaN();
        }
        else {
            // Find index for lower bound in x
            int idx = std::lower_bound(x.data(), x.data() + x.size(), xi[i]) - x.data() - 1;
            
            // Calculate fraction for interpolation
            double fraction = (xi[i] - x[idx]) / (x[idx + 1] - x[idx]);
            
            // Linear interpolation
            yi[i] = y[idx] + fraction * (y[idx + 1] - y[idx]);
        }
    }
    
    return yi;
}

