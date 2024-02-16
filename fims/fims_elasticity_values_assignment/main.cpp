#include "paramt_FIMS.h"
#include "data_inversion.h"
#include "image_processor.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <filesystem>
#include <chrono>

int main() 
{
    std::cout << "FIMS starts running... " << std::endl;  
    auto total_start = std::chrono::high_resolution_clock::now(); 
    // std::string hk_filename = "../../real_time_data/09-03-2019_Housekeeping_filtered_more.csv"; 
    std::string hk_filename = "../../raw_data/09-03-2019_Housekeeping_filtered_more.csv"; 
    std::string config_file = "../configuration/config.json";

    paramt_FIMS& fims = paramt_FIMS::getInstance(config_file);
    fims.bnd_interval = fims.data_inversion_duration / 1000;

    std::deque<House_Keeping> hk_deque;
    std::deque<std::pair<double, std::vector<cv::Point2d>>> particle_deque;
    // Mutex and condition variable for deques
    std::mutex data_mutex;
    std::condition_variable data_cv;
    std::atomic<double> frame_time(-1.0);

    int flight_num = 1;
    HK_Reader csv_reader(hk_filename, flight_num, data_mutex, data_cv, hk_deque);
    csv_reader.start();

    Data_Inversion data_inversion(data_mutex, data_cv, hk_deque, particle_deque, frame_time);  
    data_inversion.start_inversion();


    Image_Processor image_processor(flight_num, data_mutex, data_cv, particle_deque, frame_time);  
    image_processor.start();

    // std::this_thread::sleep_for(std::chrono::seconds(10));

    // image_processor.stop();
    std::cout << "image_processor stopped" << std::endl;
    data_inversion.stop_inversion();
    std::cout << "data_inversion stopped" << std::endl;
    csv_reader.stop();
    std::cout << "csv_reader stopped" << std::endl;



    auto total_stop = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_stop - total_start);
    std::cout << "FIMS finishes running... " << std::endl;    
    std::cout << "total running time: " << total_duration.count() << std::endl;
    return 0;
}