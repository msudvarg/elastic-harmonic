#include "paramt_FIMS.h"
#include "read_house_keeping.h"
#include <limits>
#include <fstream>
#include <chrono>
#include <cmath>
#include <sys/resource.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <ctime>

#define _GNU_SOURCE

void House_Keeping::set_params(int flight_num, const std::vector<double>& values) {
    UTC = values[0];
    RHa = values[4];
    Ta = values[5];
    Pabs = values[6];
    Qhs = values[7];
    Qa = values[8];
    HV = values[9];
    Pamb = Pabs / 0.95;

    switch (flight_num) {
        case 0:
            Qa_set = values[29];
            HV_set = values[30];
            break;

        case 1:
            Qa_set = values[32];
            HV_set = values[33];
            break;
    }
}

House_Keeping::House_Keeping(int flight_num, const std::vector<double>& values) {
    set_params(flight_num, values);
}
House_Keeping::House_Keeping(double time): UTC(time) {};
House_Keeping::House_Keeping(){};

HK_Reader::HK_Reader(const std::string& filename, int flight_num, std::mutex& data_mutex, std::condition_variable& data_cv, std::deque<House_Keeping>& hk_deque)
        : filename_(filename), flight_num_(flight_num), data_mutex_(data_mutex), data_cv_(data_cv), hk_deque_(hk_deque), stop_thread_(false), current_bin_time(-1.0) {}


void HK_Reader::start() {
    reader_thread_ = std::thread(&HK_Reader::read_csv2, this);
    // pin thread to cpu2
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);

    pthread_t native_thread = reader_thread_.native_handle();

    if (pthread_setaffinity_np(native_thread, sizeof(cpu_set_t), &cpuset)) {
        std::cerr << "Could not set thread to CPU 2" << std::endl;
    }
    
    struct sched_param schedParam;
    schedParam.sched_priority = 96;  // Set the priority to 96
    if (pthread_setschedparam(native_thread, SCHED_FIFO, &schedParam) != 0) {
        std::cerr << "Failed to set thread scheduling policy and priority" << std::endl;
    }
}



void HK_Reader::stop() {
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        stop_thread_ = true;
    }
    data_cv_.notify_all(); 
    reader_thread_.join();
}



void HK_Reader::read_csv() {
    std::ifstream file(filename_);
    std::string line;

    // read header
    std::getline(file, line);

    while (true) {
        if (stop_thread_) 
            break;

        if (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<double> values;

            std::string token;
            while (std::getline(iss, token, ',')) {
                try {
                    double value = std::stod(token);
                    values.push_back(value);
                } catch (const std::invalid_argument&) {
                    values.push_back(std::numeric_limits<double>::quiet_NaN());
                    std::cerr << "WARNING: Unable to parse value '" << token << "' as a double in hk file. " << std::endl;
                }
            }

            if (!values.empty()) {
                values[0] = values[0] * 3600 * 24; //convert to seconds
                if(current_bin_time == -1) {
                    current_bin_time = std::floor(values[0]);
                    cur_hk_values = values;
                    values[0] = current_bin_time;
                    House_Keeping hk(flight_num_, values);
                    std::unique_lock<std::mutex> lock(data_mutex_);
                    hk_deque_.emplace_back(hk);
                    lock.unlock();
                    data_cv_.notify_all(); 
                }
                else if(values[0] >= current_bin_time + 1) {
                    current_bin_time += 1;
                    std::vector cur_values = interp_vector(cur_hk_values, values, current_bin_time);
                    House_Keeping hk(flight_num_, cur_values);
                    std::unique_lock<std::mutex> lock(data_mutex_);
                    hk_deque_.emplace_back(hk);
                    lock.unlock();
                    data_cv_.notify_all(); 

                }
                else if(values[0] < current_bin_time + 1 && values[0] > cur_hk_values[0]) {
                    cur_hk_values = values;
                }
            }
        }
    }
}

void HK_Reader::read_csv2() {
    paramt_FIMS& fims = paramt_FIMS::getInstance();
    /************load hk file to memory**************/
    std::ifstream file_(filename_, std::ios::in | std::ios::binary);
    if (!file_.is_open()) {
        std::cerr << "Failed to open the HK file: " << filename_ << std::endl;
        return;
    }
    std::cout << "HKFile opened successfully." << std::endl;

    // Read entire content of the file into a std::string
    std::string content((std::istreambuf_iterator<char>(file_)), std::istreambuf_iterator<char>());
    file_.close();

    std::stringstream file(content);
    /************load file to memory**************/
    //barrier  
    while (!fims.finished_preloading) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); 
    }
    // std::this_thread::sleep_for(std::chrono::seconds(50));  

    std::string line;
    // read header
    std::getline(file, line);

    // std::ofstream read_time("../result_/run_time/hk_reading_time_.txt");
    // std::ofstream hk_reading_time_cpu("../result_/run_time/hk_reading_time_user.txt");
    // std::ofstream hk_reading_time_user("../result_/run_time/hk_reading_time_user.txt");
    // std::ofstream hk_reading_time_sys("../result_/run_time/hk_reading_time_sys.txt");
    // struct timespec cpustart, cpuend;
    // clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpustart); 
    // struct rusage usage_start, usage_end;
    // getrusage(RUSAGE_THREAD, &usage_start);
    auto hk_reading_duration = std::chrono::milliseconds(int(fims.HK_reading_duration));
    int loop = 0;
    double init_time = -1;
    std::vector<double> hk_reading_time(2500, 0);
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        if (stop_thread_) {
            std::ofstream read_time_("../result_/run_time/hk_reading_time.txt");
            for(int i = 0; i < hk_reading_time.size(); ++i) {
                if(hk_reading_time[i] != 0 && hk_reading_time[i + 1] != 0) {
                   read_time_ << hk_reading_time[i] << " " << std::flush; 
                }
            }
            read_time_.close();
            // read_time.close();
            break;
        }

        if (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<double> values;

            std::string token;
            while (std::getline(iss, token, ',')) {
                try {
                    double value = std::stod(token);
                    values.push_back(value);
                } catch (const std::invalid_argument&) {
                    values.push_back(std::numeric_limits<double>::quiet_NaN());
                    std::cerr << "WARNING: Unable to parse value '" << token << "' as a double in hk file. " << std::endl;
                }
            }

            if (!values.empty()) {
                //convert time to hours
                double days;
                double hours = modf(values[0], &days) * 24.0;
                if(init_time < 0) {
                    init_time = hours;
                }
                else {
                    hours = init_time + (hours - init_time)*fims.HK_reading_duration/500;
                }
                values[0] = hours;
                House_Keeping hk(flight_num_, values);
                std::unique_lock<std::mutex> lock(data_mutex_);
                hk_deque_.emplace_back(hk);
                lock.unlock();
                data_cv_.notify_all(); 
                
            }
        } 
        // getrusage(RUSAGE_THREAD, &usage_end);
        // double user_time_millis = ((usage_end.ru_utime.tv_sec - usage_start.ru_utime.tv_sec) * 1000.0) +
        //                         (usage_end.ru_utime.tv_usec - usage_start.ru_utime.tv_usec) / 1000.0;
        // double system_time_millis = ((usage_end.ru_stime.tv_sec - usage_start.ru_stime.tv_sec) * 1000.0) +
        //                             (usage_end.ru_stime.tv_usec - usage_start.ru_stime.tv_usec) / 1000.0;
        // usage_start = usage_end;
        // hk_reading_time_user << user_time_millis << " ";
        // hk_reading_time_sys << system_time_millis << " ";
        // clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpuend);
        // double elapsed = (cpuend.tv_sec - cpustart.tv_sec) * 1000 + (cpuend.tv_nsec - cpustart.tv_nsec) / 1e6;
        // hk_reading_time_cpu << elapsed << " ";
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // read_time << duration.count() << " " << std::flush;
        hk_reading_time[loop++] = duration.count();
        start += hk_reading_duration;
        std::this_thread::sleep_until(start);
        
    }
}



std::vector<double> HK_Reader::interp_vector(const std::vector<double>& v1, const std::vector<double>& v2, double value3) {
    if (v1.size() != v2.size()) {
        throw std::runtime_error("v1 and v2 must have the same size");
    }

    double x1 = v1[0];
    double x2 = v2[0];

    if (x1 == x2) {
        throw std::runtime_error("v1[0] and v2[0] must be distinct");
    }

    double t = (value3 - x1) / (x2 - x1);
    std::vector<double> result(v1.size());
    result[0] = value3;
    for (size_t i = 1; i < v1.size(); ++i) {
        result[i] = v1[i] + t * (v2[i] - v1[i]);
    }

    return result;
}

