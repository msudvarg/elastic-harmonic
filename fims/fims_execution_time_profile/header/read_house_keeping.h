#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <deque>

class House_Keeping {
public:
    //flight_independent parameters
    double UTC;
    double RHa;
    double Ta;
    double Pabs;
    double Qhs;
    double Qa;
    double HV;

    //flight_dependent parameters
    double Qa_set;
    double HV_set;
    // double V_dP; 
    double Pamb;
    // double Palt;
    //     if options.calculate_Pamb
    //     HK.Pamb = HK.Pabs / paramt_FIMS.P_ratio;
    //     if options.calculate_Palt
    //         HK.Palt = p2alt(HK.Pamb);
    House_Keeping();
    House_Keeping(double time);
    House_Keeping(int flight_num, const std::vector<double>& values);

private:
    void set_params(int flight_num, const std::vector<double>& values);


};

class HK_Reader {
public:

    // HK_Reader(const std::string& filename, int flight_num, std::mutex& data_mutex, std::condition_variable& data_cv, std::deque<House_Keeping>& hk_deque);
    HK_Reader(const std::string& filename, int flight_num, std::mutex& data_mutex, std::condition_variable& data_cv, std::deque<House_Keeping>& hk_deque);
    //start reading
    void start(); 

    //stop reading
    void stop(); 

private:
    std::string filename_;
    int flight_num_;
    std::mutex& data_mutex_;
    std::condition_variable& data_cv_;
    std::deque<House_Keeping>& hk_deque_;
    std::thread reader_thread_;
    bool stop_thread_;
    double current_bin_time;
    std::vector<double> cur_hk_values;

    void read_csv(); 
    void read_csv2();
    std::vector<double> interp_vector(const std::vector<double>& v1, const std::vector<double>& v2, double value3);
};

