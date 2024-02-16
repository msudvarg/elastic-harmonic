#include "paramt_FIMS.h"
#include "image_processor.h"

#include <iomanip>
#include <fstream>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <cmath>
#include <string>
#include <sstream>
#include <numeric>
#include <chrono>
// #include <omp.h>
// #include <boost/interprocess/shared_memory_object.hpp>
// #include <boost/interprocess/mapped_region.hpp>
// #include <boost/interprocess/sync/interprocess_mutex.hpp>
// #include <boost/interprocess/sync/scoped_lock.hpp>
// #include <boost/interprocess/sync/interprocess_condition.hpp>
// #include <boost/interprocess/managed_shared_memory.hpp>
#include <sys/resource.h>
#include <sys/time.h>

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <cstdlib>
#include <ctime>



Image_Processor::Image_Processor(int flight_num, std::mutex& data_mutex, std::condition_variable& data_cv, 
                    std::deque<std::pair<double, std::vector<cv::Point2d>>>& particle_deque, std::atomic<double>& frame_time) : 
                    flight_num_(flight_num), data_mutex_(data_mutex), data_cv_(data_cv),
                    particle_deque_(particle_deque), frame_time_(frame_time), start_time(-1), stop_thread_(false) {};

void Image_Processor::start() {
    // //ping thread to cpu2
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(2, &cpuset);
    // pthread_t current_thread = pthread_self();
    // if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset)) {
    //     std::cerr << "Could not set thread to CPU 2" << std::endl;
    // }

    // struct sched_param schedParam;
    // schedParam.sched_priority = 98;  // Set the priority to 98
    // if (pthread_setschedparam(current_thread, SCHED_FIFO, &schedParam) != 0) {
    //     std::cerr << "Failed to set thread scheduling policy and priority" << std::endl;
    // }

    fileread_n_process();
}


void Image_Processor::stop() {
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        stop_thread_ = true;
    }
    data_cv_.notify_all(); 
    processor_thread_.join();
}

void Image_Processor::fileread_n_process() {
    paramt_FIMS& fims = paramt_FIMS::getInstance();
    // std::string dir = "../../real_time_data/_00206#190904_005247";
    // std::string dir = "../../real_time_data/5stack";
    std::string dir = "../../raw_data/" + fims.test_images[fims.image_processing_duration];
    std::vector<std::string> files;
    get_files(dir, files);
    sort(files.begin(), files.end());
    std::cerr << "num images: " << files.size() << std::endl;
    // std::ofstream image_process_cost("../../result_/run_time/image_process_cost.txt");
    std::ofstream image_process_cost_user("../../result_/run_time/image_process_cost_user.txt");
    std::ofstream image_process_cost_sys("../../result_/run_time/image_process_cost_sys.txt");

    int loop = 0;
    // for(auto file : files) {
    for(int i = 0; i < files.size(); ++i) {
        // std::cerr << ++loop << std::endl;
        // if((i) % 2 == 0) continue;
        std::string file = files[i];

        auto start = std::chrono::high_resolution_clock::now();
        // struct rusage usage_start, usage_end;
        // getrusage(RUSAGE_SELF, &usage_start); 

        cv::Mat img = cv::imread(file);
        if(img.empty()) {
            std::cerr << "Cannot read input image." << std::endl;
            return;
        }
        // std::string timestamp = file.stem().string();
        std::filesystem::path filePath(file);
        std::string timestamp = filePath.stem().string();
        img.at<uchar>(275, 135) = 0;
        double t = convert_time2hours(timestamp);
        image_process(img, t);

        // getrusage(RUSAGE_SELF, &usage_end); 
        // double user_time_millis = ((usage_end.ru_utime.tv_sec - usage_start.ru_utime.tv_sec) * 1000.0) + 
        //                         (usage_end.ru_utime.tv_usec - usage_start.ru_utime.tv_usec) / 1000.0;
        // double system_time_millis = ((usage_end.ru_stime.tv_sec - usage_start.ru_stime.tv_sec) * 1000.0) + 
        //                             (usage_end.ru_stime.tv_usec - usage_start.ru_stime.tv_usec) / 1000.0;

        // image_process_cost_user << user_time_millis << " ";
        // image_process_cost_sys << system_time_millis << " ";

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // image_process_cost << duration.count() << " ";

        // if(duration < std::chrono::milliseconds(100))
        //     std::this_thread::sleep_for(std::chrono::milliseconds(100) - duration);

    }
    image_process_cost_user.close();
    image_process_cost_sys.close();
    // std::cout << "writing result to local file, please wait..." << std::endl;

    // for(int i = 0; i < 2000; ++i) {
    //     data_cv_.notify_all(); 
    //     std::this_thread::sleep_for(std::chrono::milliseconds(15));
    // }

}

/*
// Use ZMQ for raw data transmitting
void Image_Processor::zmqread_n_process() {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10); 
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::sub);  // SUB socket type for subscribing to messages

    // Connect to the socket
    socket.connect("tcp://localhost:5555");

    // Subscribe to all messages
    socket.set(zmq::sockopt::subscribe, "");

    while (true) {
        zmq::message_t timestamp_msg;
        if (!socket.recv(timestamp_msg, zmq::recv_flags::none)) {
            std::cerr << "Error receiving message" << std::endl;
        }
        
        std::string timestamp(static_cast<char*>(timestamp_msg.data()), timestamp_msg.size());
        double t = convert_time2hours(timestamp);
        
        // The second part of the message is the image data
        zmq::message_t image_msg;
        if (!socket.recv(image_msg, zmq::recv_flags::none)) {
            std::cerr << "Error receiving message" << std::endl;
        }
        std::vector<uchar> image_data(static_cast<char*>(image_msg.data()), static_cast<char*>(image_msg.data()) + image_msg.size());

        cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
        image_process(img, t);

    }
}

//Shared memory for raw data transmitting
void Image_Processor::shmread_n_process() {
    const int BUFFER_SIZE = 100;
    const int HEIGHT = 1024;
    const int WIDTH = 196;
    const int CHANNEL = 1;
    const int MAX_BUFFER_SIZE = HEIGHT * WIDTH * CHANNEL * sizeof(uchar) + sizeof(double);
    // Remove existing shared memory
    boost::interprocess::shared_memory_object::remove("fimsSharedMemory");   
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        std::ostringstream name_stream;
        name_stream << "image_shm" << i;
        std::string name = name_stream.str();
        boost::interprocess::shared_memory_object::remove(name.c_str());
    }

    // Create shared memory to hold the write and read indices
    boost::interprocess::managed_shared_memory segment(boost::interprocess::open_or_create, "fimsSharedMemory", 65536); 
    int* write_index = segment.find_or_construct<int>("write_index")(0);
    int* read_index = segment.find_or_construct<int>("read_index")(0);

    // Create shared condition variable and mutex
    boost::interprocess::interprocess_mutex* mutex = segment.find_or_construct<boost::interprocess::interprocess_mutex>("mutex")();
    boost::interprocess::interprocess_condition* cond = segment.find_or_construct<boost::interprocess::interprocess_condition>("cond")();

    boost::interprocess::shared_memory_object shm[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        std::ostringstream name_stream;
        name_stream << "image_shm" << i;
        std::string name = name_stream.str();
        try {
            shm[i] = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, name.c_str(), boost::interprocess::read_write);
            shm[i].truncate(MAX_BUFFER_SIZE);
        }
        catch (boost::interprocess::interprocess_exception& ex) {
            std::cerr << "Share memory allocating exception: " << ex.what() << std::endl;
        }

    }


    while (true) {
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(*mutex);
        while (*write_index == *read_index) {
            cond->wait(lock);  // Wait if the buffer is empty
        }

        // Generate shared memory block name
        std::ostringstream name_stream;
        name_stream << "image_shm" << *read_index;
        std::string name = name_stream.str();
        cv::Mat img;
        double timestamp;
        try {
            // Open the shared memory block
            boost::interprocess::shared_memory_object shm(boost::interprocess::open_only, name.c_str(), boost::interprocess::read_only);

            // Map the whole shared memory in this process
            boost::interprocess::mapped_region region(shm, boost::interprocess::read_only);

            // Get the region address
            void* addr = region.get_address();
            
            // Construct the image timestamp and data from the shared memory block
            memcpy(&timestamp, addr, sizeof(double));
            cv::Mat img_original(HEIGHT, WIDTH, CV_8UC1, static_cast<char*>(addr) + sizeof(double));
            cv::cvtColor(img_original, img, cv::COLOR_GRAY2BGR);

            *read_index = (*read_index + 1) % BUFFER_SIZE;
            lock.unlock();
            cond->notify_one();  // Notify the producer that a new buffer slot is available
        }
        catch(boost::interprocess::interprocess_exception& ex) {
            std::cerr << "Failed to open shared memory block " << name << ": " << ex.what() << std::endl;
            return;
        }
        

        double t = timestamp / 24 / 3600;
        double days;
        double hours  = modf(t, &days) * 24.0;
        image_process(img, hours);
    }

}
*/

void Image_Processor::image_process(cv::Mat& img, double t) {

    paramt_FIMS& fims = paramt_FIMS::getInstance();
    cv::Mat imgGray = img;
    //convert image to gray scale
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    blob_removal(imgGray, fims.pix_lim);

    cv::Vec4d hv_wall = channel_wall_positions_from_image(imgGray, fims.pxlim_hv, fims.pxlim_gnd);
    cv::Mat particles = particle_detection (imgGray, fims.thresh_arr, fims.area_arr, fims.pix_lim);
    cv::Mat converted_particles = convert_coordinates (particles, hv_wall, fims.hv_px0, fims.gnd_px0, fims.hv_slope, fims.gnd_slope, fims.pyc, fims.a);
    cv::Mat filtered_particles = filter_particles(converted_particles, fims.x_min, fims.x_max, fims.y_min, fims.y_max);

    if(start_time < 0) {
        start_time = t;
        // next_bin_time = t;
        // next_bin_time = 0.879722; //for dataset 206
        next_bin_time = 0.629999999888241; //for dataset 191
        
    }
    else {
        // calibrate time
        // t = start_time + (t - start_time) * 0.9996678369563557; //for dataset 206
        t = start_time + (t - start_time) * 0.9987719235731375; //for dataset 191
    }

    sort_particles (filtered_particles, t, fims.Qt_set, fims.a, fims.b, fims.Lc, fims.bnd_interval);
    frame_time_ = t;

}

cv::Mat Image_Processor::particle_detection(cv::Mat& ig, std::vector<int>& thresh_arr, std::vector<int>& area_arr, std::vector<int>& pix_lim) {
    if(!pix_lim.empty()) {
        ig = ig(cv::Range::all(), cv::Range(pix_lim[0], pix_lim[1]));
    }

    std::vector<int> dupcount(thresh_arr.size(), 0);
    std::vector<std::pair<double, double>> this_frame_coordinates;
    std::vector<int> this_frame_maxvals;
    std::vector<int> this_frame_areas;

    for(size_t thresh_idx = 0; thresh_idx < thresh_arr.size(); ++thresh_idx) {
        if(thresh_idx < thresh_arr.size()-1) {
            if(thresh_arr[thresh_idx] < thresh_arr[thresh_idx+1]) {
                std::cerr << "Thresholds out of order!" << std::endl;
            }
        }
        int thresh = thresh_arr[thresh_idx]-1;

        // Apply threshold
        cv::Mat BW;
        const int max_value = 255; 
        cv::threshold(ig, BW, thresh, max_value, cv::THRESH_BINARY);

        // Get regions and properties
        const int connectivity = 8;
        // cv::Mat labels, stats, centroids;
        int numObj = cv::connectedComponentsWithStats(BW, labels, stats, centroids, connectivity);
        // Locating x and y positions
        std::vector<std::pair<double, double>> coordinates;
        // Find weighted centroids for each component
        std::vector<std::pair<double, double>> weighted_coordinates;
        // Find areas of each region
        std::vector<int> areas;
        // Find max intensity for each component
        std::vector<int> max_val;
        compute_weighted_centroids(numObj, centroids, stats, labels, ig, coordinates, weighted_coordinates, areas, max_val);

        // Check for duplicate regions
        std::vector<int> new_reg_idx;
        if(thresh_idx > 0) {
            for(size_t label_idx = 1; label_idx < numObj; ++label_idx) {
                if(max_val[label_idx] < thresh_arr[thresh_idx-1] && areas[label_idx] >= area_arr[thresh_idx]) {
                    new_reg_idx.push_back(label_idx);
                    this_frame_coordinates.push_back(weighted_coordinates[label_idx]);
                    this_frame_maxvals.push_back(max_val[label_idx]);
                    this_frame_areas.push_back(areas[label_idx]);
                }
            }
        }
        else{
            for(size_t label_idx = 1; label_idx < numObj; ++label_idx) {
                if(areas[label_idx] >= area_arr[thresh_idx]) {
                    new_reg_idx.push_back(label_idx);
                    this_frame_coordinates.push_back(weighted_coordinates[label_idx]);
                    this_frame_maxvals.push_back(max_val[label_idx]);
                    this_frame_areas.push_back(areas[label_idx]);
                }
            }
        }
        dupcount[thresh_idx] = new_reg_idx.size();
    }

    // Removal of double counting
    std::vector<std::pair<double, double>> this_frame_coordinates_final;
    std::vector<int> this_frame_maxvals_final;
    std::vector<int> this_frame_areas_final;
    remove_double_counting(this_frame_coordinates, this_frame_coordinates_final, this_frame_maxvals_final, this_frame_areas_final, this_frame_maxvals, this_frame_areas, pix_lim);

    cv::Mat result(this_frame_coordinates_final.size(), 2, CV_64F);
    for(int i = 0; i < this_frame_coordinates_final.size(); ++i) {
        result.at<double>(i, 0) = this_frame_coordinates_final[i].first;
        result.at<double>(i, 1) = this_frame_coordinates_final[i].second;
    }
    return result;
}

// // parallelise using openMP
// cv::Mat Image_Processor::particle_detection_par1 (cv::Mat& ig, std::vector<int>& thresh_arr, std::vector<int>& area_arr, std::vector<int>& pix_lim) {
//     if(!pix_lim.empty()) {
//         ig = ig(cv::Range::all(), cv::Range(pix_lim[0], pix_lim[1]));
//     }

//     std::vector<int> dupcount(thresh_arr.size(), 0);
//     std::vector<std::vector<std::pair<double, double>>> this_frame_coordinates(thresh_arr.size());
//     std::vector<std::vector<int>> this_frame_maxvals(thresh_arr.size());
//     std::vector<std::vector<int>> this_frame_areas(thresh_arr.size());

//     #pragma omp parallel for
//     for(size_t thresh_idx = 0; thresh_idx < thresh_arr.size(); ++thresh_idx) {
//         if(thresh_idx < thresh_arr.size()-1) {
//             if(thresh_arr[thresh_idx] < thresh_arr[thresh_idx+1]) {
//                 std::cerr << "Thresholds out of order!" << std::endl;
//             }
//         }
//         int thresh = thresh_arr[thresh_idx]-1;

//         // Apply threshold
//         cv::Mat BW;
//         const int max_value = 255;
//         cv::threshold(ig, BW, thresh, max_value, cv::THRESH_BINARY);

//         // Get regions and properties
//         const int connectivity = 8;
//         cv::Mat labels, stats, centroids;
//         int numObj = cv::connectedComponentsWithStats(BW, labels, stats, centroids, connectivity);
//         // Locating x and y positions
//         std::vector<std::pair<double, double>> coordinates;

//         // Find weighted centroids for each component
//         std::vector<std::pair<double, double>> weighted_coordinates;

//         // Find areas of each region
//         std::vector<int> areas;

//         // Find max intensity for each component
//         std::vector<int> max_val;

//         compute_weighted_centroids(numObj, centroids, stats, labels, ig, coordinates, weighted_coordinates, areas, max_val);

//         // Check for duplicate regions
//         if(thresh_idx > 0) {
//             for(size_t label_idx = 1; label_idx < numObj; ++label_idx) {
//                 if(max_val[label_idx] < thresh_arr[thresh_idx-1] && areas[label_idx] >= area_arr[thresh_idx]) {
//                     this_frame_coordinates[thresh_idx].push_back(weighted_coordinates[label_idx]);
//                     this_frame_maxvals[thresh_idx].push_back(max_val[label_idx]);
//                     this_frame_areas[thresh_idx].push_back(areas[label_idx]);
//                 }
//             }
//         }
//         else{
//             for(size_t label_idx = 1; label_idx < numObj; ++label_idx) {
//                 if(areas[label_idx] >= area_arr[thresh_idx]) {
//                     this_frame_coordinates[thresh_idx].push_back(weighted_coordinates[label_idx]);
//                     this_frame_maxvals[thresh_idx].push_back(max_val[label_idx]);
//                     this_frame_areas[thresh_idx].push_back(areas[label_idx]);
//                 }
//             }
//         }
//     }

//     // Removal of double counting
//     std::vector<std::pair<double, double>> this_frame_coordinates_final;
//     std::vector<int> this_frame_maxvals_final;
//     std::vector<int> this_frame_areas_final;
//     remove_double_counting_par(this_frame_coordinates, this_frame_coordinates_final, this_frame_maxvals_final, this_frame_areas_final, this_frame_maxvals, this_frame_areas, pix_lim);

//     cv::Mat result(this_frame_coordinates_final.size(), 2, CV_64F);
//     for(int i = 0; i < this_frame_coordinates_final.size(); ++i) {
//         result.at<double>(i, 0) = this_frame_coordinates_final[i].first;
//         result.at<double>(i, 1) = this_frame_coordinates_final[i].second;
//     }
//     return result;
// }

// parallelise using std::thread
cv::Mat Image_Processor::particle_detection_par(cv::Mat& ig, std::vector<int>& thresh_arr, std::vector<int>& area_arr, std::vector<int>& pix_lim) {
    if(!pix_lim.empty()) {
        ig = ig(cv::Range::all(), cv::Range(pix_lim[0], pix_lim[1]));
    }

    std::vector<int> dupcount(thresh_arr.size(), 0);
    std::vector<std::vector<std::pair<double, double>>> this_frame_coordinates(thresh_arr.size());
    std::vector<std::vector<int>> this_frame_maxvals(thresh_arr.size());
    std::vector<std::vector<int>> this_frame_areas(thresh_arr.size());


    std::vector<std::thread> threads;
    for(size_t thresh_idx = 0; thresh_idx < thresh_arr.size(); ++thresh_idx) {
        threads.push_back(std::thread([&, thresh_idx](){
            if(thresh_idx < thresh_arr.size()-1) {
                if(thresh_arr[thresh_idx] < thresh_arr[thresh_idx+1]) {
                    std::cerr << "Thresholds out of order!" << std::endl;
                }
            }
            int thresh = thresh_arr[thresh_idx]-1;

            // Apply threshold
            cv::Mat BW;
            const int max_value = 255;
            cv::threshold(ig, BW, thresh, max_value, cv::THRESH_BINARY);

            // Get regions and properties
            const int connectivity = 8;
            cv::Mat labels, stats, centroids;
            int numObj = cv::connectedComponentsWithStats(BW, labels, stats, centroids, connectivity);
            // Locating x and y positions
                       
            std::vector<std::pair<double, double>> coordinates;

            // Find weighted centroids for each component
            std::vector<std::pair<double, double>> weighted_coordinates;

            // Find areas of each region
            std::vector<int> areas;

            // Find max intensity for each component
            std::vector<int> max_val;
            compute_weighted_centroids(numObj, centroids, stats, labels, ig, coordinates, weighted_coordinates, areas, max_val);

            // Check for duplicate regions
            if(thresh_idx > 0) {
                for(size_t label_idx = 1; label_idx < numObj; ++label_idx) {
                    if(max_val[label_idx] < thresh_arr[thresh_idx-1] && areas[label_idx] >= area_arr[thresh_idx]) {
                        this_frame_coordinates[thresh_idx].push_back(weighted_coordinates[label_idx]);
                        this_frame_maxvals[thresh_idx].push_back(max_val[label_idx]);
                        this_frame_areas[thresh_idx].push_back(areas[label_idx]);
                    }
                }
            }
            else{
                for(size_t label_idx = 1; label_idx < numObj; ++label_idx) {
                    if(areas[label_idx] >= area_arr[thresh_idx]) {
                        this_frame_coordinates[thresh_idx].push_back(weighted_coordinates[label_idx]);
                        this_frame_maxvals[thresh_idx].push_back(max_val[label_idx]);
                        this_frame_areas[thresh_idx].push_back(areas[label_idx]);
                    }
                }
            }

        }));
    }
    for(auto& thread : threads) {
        thread.join();
    }

    // Removal of double counting
    std::vector<std::pair<double, double>> this_frame_coordinates_final;
    std::vector<int> this_frame_maxvals_final;
    std::vector<int> this_frame_areas_final;
    remove_double_counting_par(this_frame_coordinates, this_frame_coordinates_final, this_frame_maxvals_final, this_frame_areas_final, this_frame_maxvals, this_frame_areas, pix_lim);

    cv::Mat result(this_frame_coordinates_final.size(), 2, CV_64F);
    for(int i = 0; i < this_frame_coordinates_final.size(); ++i) {
        result.at<double>(i, 0) = this_frame_coordinates_final[i].first;
        result.at<double>(i, 1) = this_frame_coordinates_final[i].second;
    }
    return result;
}


void Image_Processor::remove_double_counting_par(std::vector<std::vector<std::pair<double, double>>>& this_frame_coordinates,
                            std::vector<std::pair<double, double>>& this_frame_coordinates_final,
                            std::vector<int>& this_frame_maxvals_final,
                            std::vector<int>& this_frame_areas_final,
                            std::vector<std::vector<int>>& this_frame_maxvals,
                            std::vector<std::vector<int>>& this_frame_areas,
                            std::vector<int>& pix_lim) {
    
    int size = std::accumulate(this_frame_coordinates.begin(), this_frame_coordinates.end(), 0, [](int count, const auto& innerVector) {
        return count + innerVector.size();
    });

    std::vector<std::pair<double, double>> coordinates;
    coordinates.reserve(size);
    for (const auto& innerVector : this_frame_coordinates) {
        coordinates.insert(coordinates.end(), innerVector.begin(), innerVector.end());
    }

    std::vector<int> maxvals;
    maxvals.reserve(size);       
    for (const auto& innerVector : this_frame_maxvals) {
        maxvals.insert(maxvals.end(), innerVector.begin(), innerVector.end());
    }

    std::vector<int> areas;
    areas.reserve(size);
    for (const auto& innerVector : this_frame_areas) {
        areas.insert(areas.end(), innerVector.begin(), innerVector.end());
    }
    
    // Removal of double counting
    std::unordered_set<int> double_cnt_line;
    int dbl_cnt_pix = 3;
    if(coordinates.size() > 1) {
        for(int j = 0; j < coordinates.size(); ++j) {
            for(int i = j+1; i < coordinates.size(); ++i) {
                double x_diff = coordinates[i].first-coordinates[j].first;
                double y_diff = coordinates[i].second-coordinates[j].second;
                int K = std::sqrt(x_diff*x_diff + y_diff*y_diff); 
                if(K < dbl_cnt_pix) {
                    double_cnt_line.insert(j);
                    break;
                }
            }
        }
    }
 
    // Start removal
    for(int i = 0; i < coordinates.size(); ++i) {
        if(double_cnt_line.find(i) == double_cnt_line.cend()) {
            this_frame_areas_final.push_back(areas[i]);
            this_frame_coordinates_final.push_back(std::pair<double, double>(coordinates[i].first+pix_lim[0], coordinates[i].second));
            this_frame_maxvals_final.push_back(maxvals[i]);
        }
    }
}


void Image_Processor::remove_double_counting(std::vector<std::pair<double, double>>& this_frame_coordinates,
                            std::vector<std::pair<double, double>>& this_frame_coordinates_final,
                            std::vector<int>& this_frame_maxvals_final,
                            std::vector<int>& this_frame_areas_final,
                            std::vector<int>& this_frame_maxvals,
                            std::vector<int>& this_frame_areas,
                            std::vector<int>& pix_lim) {
    // Removal of double counting
    std::unordered_set<int> double_cnt_line;
    int dbl_cnt_pix = 3;
    if(this_frame_coordinates.size() > 1) {
        for(int j = 0; j < this_frame_coordinates.size(); ++j) {
            for(int i = j+1; i < this_frame_coordinates.size(); ++i) {
                double x_diff = this_frame_coordinates[i].first-this_frame_coordinates[j].first;
                double y_diff = this_frame_coordinates[i].second-this_frame_coordinates[j].second;
                int K = std::sqrt(x_diff*x_diff + y_diff*y_diff); 
                if(K < dbl_cnt_pix) {
                    double_cnt_line.insert(j);
                    break;
                }
            }
        }
    }
    // Start removal
    for(int i = 0; i < this_frame_coordinates.size(); ++i) {
        if(double_cnt_line.find(i) == double_cnt_line.cend()) {
            this_frame_areas_final.push_back(this_frame_areas[i]);
            this_frame_coordinates_final.push_back(std::pair<double, double>(this_frame_coordinates[i].first+pix_lim[0], this_frame_coordinates[i].second));
            this_frame_maxvals_final.push_back(this_frame_maxvals[i]);
        }
    }
}


void Image_Processor::compute_weighted_centroids(int numObj, cv::Mat& centroids, cv::Mat& stats, cv::Mat& labels, cv::Mat& ig,
                                std::vector<std::pair<double, double>>& coordinates,
                                std::vector<std::pair<double, double>>& weighted_coordinates,
                                std::vector<int>& areas,
                                std::vector<int>& max_val) {
    cv::Mat_<uchar> ig_mat = ig.isContinuous()? ig : ig.clone();
    cv::Mat_<int> labels_mat = labels.isContinuous()? labels : labels.clone();

    for(size_t label_idx = 0; label_idx < numObj; ++label_idx) {
        std::pair<double, double> coordinate(centroids.at<double>(label_idx, 0), centroids.at<double>(label_idx, 1));
        coordinates.push_back(coordinate);
        int area = stats.at<int>(label_idx, cv::CC_STAT_AREA);
        areas.push_back(area);
        const int left = stats.at<int>(label_idx, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label_idx, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label_idx, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label_idx, cv::CC_STAT_HEIGHT);
        double x_sum = 0;
        double y_sum = 0;
        double w_sum = 0;
        int max_intensity = -1;
        for (int y_coordinate = top; y_coordinate < top + height; ++y_coordinate) {
            for (int x_coordinate = left; x_coordinate < left + width; ++x_coordinate) {
                int index = y_coordinate * ig.cols + x_coordinate;
                if (labels_mat(index) == label_idx) {
                    int intensity = ig_mat(index);
                    w_sum += intensity;
                    x_sum += intensity * (x_coordinate);
                    y_sum += intensity * (y_coordinate);
                    if (intensity > max_intensity) {
                        max_intensity = intensity;
                    }
                }
            }
        }
        std::pair<double, double> weighted_pair(x_sum / w_sum, y_sum / w_sum);
        weighted_coordinates.push_back(weighted_pair);
        max_val.push_back(max_intensity);
    }
}

void Image_Processor::compute_weighted_centroids_par(int numObj, cv::Mat& centroids, cv::Mat& stats, cv::Mat& labels, cv::Mat& ig,
                                std::vector<std::pair<double, double>>& coordinates,
                                std::vector<std::pair<double, double>>& weighted_coordinates,
                                std::vector<int>& areas,
                                std::vector<int>& max_val) {
    cv::Mat_<uchar> ig_mat = ig.isContinuous()? ig : ig.clone();
    cv::Mat_<int> labels_mat = labels.isContinuous()? labels : labels.clone();

    std::vector<std::thread> threads;
    for(size_t label_idx = 0; label_idx < numObj; ++label_idx) {
        threads.push_back(std::thread([&, label_idx](){
            std::pair<double, double> coordinate(centroids.at<double>(label_idx, 0), centroids.at<double>(label_idx, 1));
            coordinates[label_idx] = coordinate;
            int area = stats.at<int>(label_idx, cv::CC_STAT_AREA);
            areas[label_idx] = area;
            const int left = stats.at<int>(label_idx, cv::CC_STAT_LEFT);
            const int top = stats.at<int>(label_idx, cv::CC_STAT_TOP);
            const int width = stats.at<int>(label_idx, cv::CC_STAT_WIDTH);
            const int height = stats.at<int>(label_idx, cv::CC_STAT_HEIGHT);
            double x_sum = 0;
            double y_sum = 0;
            double w_sum = 0;
            int max_intensity = -1;
            for (int y_coordinate = top; y_coordinate < top + height; ++y_coordinate) {
                for (int x_coordinate = left; x_coordinate < left + width; ++x_coordinate) {
                    int index = y_coordinate * ig.cols + x_coordinate;
                    if (labels_mat(index) == label_idx) {
                        int intensity = ig_mat(index);
                        w_sum += intensity;
                        x_sum += intensity * (x_coordinate);
                        y_sum += intensity * (y_coordinate);
                        if (intensity > max_intensity) {
                            max_intensity = intensity;
                        }
                    }
                }
            }
            std::pair<double, double> weighted_pair(x_sum / w_sum, y_sum / w_sum);
            weighted_coordinates[label_idx] = weighted_pair;
            max_val[label_idx] = max_intensity;
        }));
    }
    for(auto& thread : threads) {
        thread.join();
    }
}


cv::Vec4d Image_Processor::find_wall(cv::Mat ig, int offset) {
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    std::vector<double> px(ig.rows, NaN);
    std::vector<double> py(ig.rows);
    std::iota(py.begin(), py.end(), 0);
    std::vector<uchar> Imax(ig.rows, 0);
    std::vector<uchar> Imin(ig.rows, UCHAR_MAX);

    for (int j1 = 0; j1 < ig.rows; ++j1) {
        int sum = 0;
        int count = 0;
        uchar maxVal = 0;
        for (int i = 0; i < ig.cols; ++i) {
            uchar val = ig.at<uchar>(j1, i);
            if (val > maxVal) {
                maxVal = val;
                sum = i;
                count = 1;
            } else if (val == maxVal) {
                sum += i;
                ++count;
            }
            Imin[j1] = std::min(val, Imin[j1]);
        }
        if (count > 0) { // Avoid division by zero
            px[j1] = static_cast<double>(sum) / count + offset;
            Imax[j1] = maxVal;
        }
    }

    int idx_count = 0;
    for(int idx = 0; idx < Imax.size(); ++idx) {
        if(Imax[idx] == Imin[idx]) {
            px[idx] = NaN;
            py[idx] = NaN;
            ++idx_count;
        }
    }

    double px_max = -1;
    for(int i = 0; i < px.size(); ++i) {
        if(px[i] > px_max) {
            px_max = px[i];
        }
    }
    if(px_max < 100) {
        idx_count = 0;
        // Initialize edges, midpt
        std::vector<double> edges;
        std::vector<double> midpt;
        for (double edge = 20; edge <= 40; ++edge) {
            edges.push_back(edge);
            if(edge != 20) {
                midpt.push_back((edge+edge-1)/2);
            }
        }
        // Perform histcounts
        std::vector<int> y(edges.size()-1, 0);
        for (int j = 0; j < px.size(); ++j) {
            double x = px[j];
            for(int i = 1; i < edges.size(); ++i) {
                // Test whether x lies within the range
                if(x >= edges[i-1] && x < edges[i]) {
                    y[i-1] += 1;
                    break;
                }
            }
        }
        // Compute px_most
        double sum = 0;
        double num = 0;
        double y_max = -1;
        for (int i = 0; i < y.size(); ++i) {
            if(y[i] > y_max) {
                y_max = y[i];
            }
        }
        for (int i = 0; i < y.size(); ++i) {
            if(y[i] == y_max) {
                sum += midpt[i];
                ++num;
            }
        }
        double px_most = sum / num;
        // Compute the indices that need to be NaN
        for (int i = 0; i < px.size(); ++i) {
            double x = px[i];
            if (x > px_most+1 or x < px_most-1 or x == NaN) {
                ++idx_count;
                px[i] = NaN;
                py[i] = NaN;
            }
        }
    }

    cv::Vec4d line(NaN, NaN, NaN, NaN);
    if(idx_count <= ig.rows-2) {
        std::vector<std::vector<double>> pointsVector;
        for (int i = 0; i < ig.rows; ++i) {
            if (!std::isnan(px[i])) {
                pointsVector.push_back({px[i], py[i]});
            }
        }
        cv::Mat points(pointsVector.size(), 2, CV_64F);
        for (int i = 0; i < pointsVector.size(); ++i) {
            points.at<double>(i, 0) = pointsVector[i][0];
            points.at<double>(i, 1) = pointsVector[i][1];
        }

        cv::fitLine(points, line, cv::DIST_L2, 0, 0.001, 0.001);
        double x0 = line[2] - line[0]*line[3]/line[1];
        double y0 = 0;
        line[2] = x0;
        line[3] = y0;
    }
    return line;
}

cv::Vec4d Image_Processor::channel_wall_positions_from_image(cv::Mat& ig, std::vector<int>& pxlim_hv, std::vector<int>& pxlim_gnd) {
    int offset = 0;
    cv::Mat temp_ig;
    if(pxlim_hv.size() == 1) {
        temp_ig = ig(cv::Range::all(), cv::Range(0, pxlim_hv[0]));
        offset = 0;
    }
    else{
        temp_ig = ig(cv::Range::all(), cv::Range(pxlim_hv[0], pxlim_hv[1]));
        offset = pxlim_hv[0];
    }
    return find_wall(temp_ig, offset);
}


cv::Mat Image_Processor::convert_coordinates (cv::Mat& particles, cv::Vec4d& wall, std::pair<double, double>& hv_px0, std::pair<double, double>& gnd_px0, 
    const std::pair<double, double>& hv_slope, const std::pair<double, double>& gnd_slope, const std::pair<double, double>& pixel_yc, double a) 
{
    std::pair<double, double> hv_alpha = {atan(hv_slope.first), atan(hv_slope.second)};
    std::pair<double, double> gnd_alpha = {atan(gnd_slope.first), atan(gnd_slope.second)};

    double px0_HV = wall[2]+1 - wall[0]/wall[1]*(wall[3]+1);
    double px0_GND = (gnd_px0.second - gnd_px0.first) / (hv_px0.second - hv_px0.first) * (px0_HV - hv_px0.first) + gnd_px0.first;
    double alpha_HV = (hv_alpha.second - hv_alpha.first) / (hv_px0.second - hv_px0.first) * (px0_HV - hv_px0.first) + hv_alpha.first;
    double alpha_GND = (gnd_alpha.second - gnd_alpha.first) / (hv_px0.second - hv_px0.first) * (px0_HV - hv_px0.first) + gnd_alpha.first;
    double y_center = (pixel_yc.second - pixel_yc.first) / (hv_px0.second - hv_px0.first) * (px0_HV - hv_px0.first) + pixel_yc.first;

    double px0_wall[] = {px0_HV, px0_GND};
    double alpha_arr[] = {alpha_HV, alpha_GND};
    double x_center = (px0_wall[0] + px0_wall[1]) / 2.0 + (tan(alpha_arr[0]) + tan(alpha_arr[1])) / 2.0 * y_center;
    double alpha_m = (alpha_arr[0] + alpha_arr[1]) / 2.0;

    double factor = a / (px0_wall[1] - px0_wall[0] + y_center * (tan(alpha_arr[1]) - tan(alpha_arr[0]))) / cos(alpha_m);

    cv::Mat converted_particles(particles.rows, particles.cols, CV_64F);
    for (int i = 0; i < particles.rows; i++) {

        double p_x = particles.at<double>(i, 0) + 1 - x_center;
        double p_y = particles.at<double>(i, 1) + 1 - y_center;

        double px1 = p_x * cos(alpha_m) - p_y * sin(alpha_m);
        double py1 = p_y * cos(alpha_m) + p_x * sin(alpha_m);

        double x = a / 2.0 - px1 * factor;
        double y = py1 * factor;

        // assign x and y to output
        converted_particles.at<double>(i, 0) = x;
        converted_particles.at<double>(i, 1) = y;

    }
    return converted_particles;
}


cv::Mat Image_Processor::filter_particles(const cv::Mat& particles, double xmin, double xmax, double ymin, double ymax) {
    cv::Mat filtered_particles;
    for (int i = 0; i < particles.rows; ++i) {
        double x = particles.at<double>(i, 0);
        double y = particles.at<double>(i, 1);
        if(x >= xmin && x <= xmax && y >= ymin && y <= ymax)
            filtered_particles.push_back(particles.row(i));
    }
    return filtered_particles;
}


cv::Mat Image_Processor::correct_particle_time(double t, const cv::Mat& particles, double Qt, double a, double b, double Lc) {
    cv::Mat time = cv::Mat::ones(particles.rows, 1, CV_64F) * t;

    cv::Mat pos_x = particles.col(0);
    cv::Mat pos_x_normalized = pos_x / a;
    cv::Mat pos_x_normalized_square = pos_x_normalized.mul(pos_x_normalized);
    cv::Mat pos_x_normalized_cube = pos_x_normalized_square.mul(pos_x_normalized);

    cv::Mat ts = Qt * (3 * pos_x_normalized_square - 2 * pos_x_normalized_cube) / 3600;

    cv::Mat tc = Lc / (6 * Qt / (std::pow(a, 3) * b) * (pos_x.mul(a - pos_x))) / 3600;

    cv::Mat residence_time = ts + tc;

    cv::Mat corrected_time = time - residence_time;
    return corrected_time;
}


void Image_Processor::put_to_bin(cv::Mat& particles, cv::Mat& time, double interval) {
                
    std::deque<std::pair<double, std::vector<cv::Point2d>>>::iterator it;

    for(int i = 0; i < particles.rows; ++i) {
        double current_time = time.at<double>(i);
        {
            std::unique_lock<std::mutex> lock(data_mutex_); // Lock particle_bin with a unique_lock

            it = std::upper_bound(particle_deque_.begin(), particle_deque_.end(), current_time,
                        [](double time, const std::pair<double, std::vector<cv::Point2d>>& bin) {
                            return time < bin.first;
                        });

            if(particle_deque_.empty()){
                particle_deque_.push_back({next_bin_time, std::vector<cv::Point2d>{}});
                next_bin_time += interval/3600;
                if(current_time < particle_deque_.begin()->first)
                    continue;
                while( current_time >= particle_deque_.rbegin()->first) {
                    particle_deque_.push_back({next_bin_time, std::vector<cv::Point2d>{}});
                    next_bin_time += interval/3600;
                }
                std::prev(particle_deque_.end(), 2)->second.push_back(particles.at<cv::Point2d>(i));    
            }
            else if (it == particle_deque_.begin()) {
                continue;
            }
            else if (it != particle_deque_.end()) {
                std::prev(it)->second.push_back(particles.at<cv::Point2d>(i));        
            } 
            else if(current_time >= particle_deque_.rbegin()->first){
                while( current_time >= particle_deque_.rbegin()->first) {
                    particle_deque_.push_back({next_bin_time, std::vector<cv::Point2d>{}});
                    next_bin_time += interval/3600;
                }
                std::prev(particle_deque_.end(), 2)->second.push_back(particles.at<cv::Point2d>(i));    
            }

            lock.unlock(); 
            data_cv_.notify_all(); 
        }
    }
}

void Image_Processor::sort_particles (cv::Mat& particles, double t, double Qt, double a, 
                                        double b, double Lc, double interval) {

    if (particles.empty()) {
        return;
    }

    if (t == -1) {
        std::cerr << "Error: time not valid." << std::endl;
        return;
    }

    cv::Mat time = correct_particle_time(t, particles, Qt, a, b, Lc);
    put_to_bin(particles, time, interval);
}

void Image_Processor::get_files(const std::string& dir, std::vector<std::string>& files) {
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".png") {
            files.push_back(entry.path());
        }
    }    
}

double Image_Processor::convert_time2hours (std::string & timestamp) {
    double t;
    try {
        t = std::stod(timestamp) / 24 / 3600;
    } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << '\n';
    } catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << '\n';
    }
    double days;
    double hours = modf(t, &days) * 24.0;
    return hours;
}

void Image_Processor::blob_removal(cv::Mat& img, std::vector<int>& particle_px_lim) {
    // Define the ranges for pxlim_hv and pxlim_gnd

    std::pair<int, int> pxlim_hv = {particle_px_lim[0]-3, particle_px_lim[0]-1};  
    std::pair<int, int> pxlim_gnd = {particle_px_lim[1]-4, particle_px_lim[1]-2}; 

    cv::Mat tmp_ig_hv = img(cv::Range::all(), cv::Range(pxlim_hv.first, pxlim_hv.second));
    cv::Mat tmp_ig_gnd = img(cv::Range::all(), cv::Range(pxlim_gnd.first, pxlim_gnd.second));

    // Calculate maximum values for each row
    cv::Mat Imax_hv, Imax_gnd;
    cv::reduce(tmp_ig_hv, Imax_hv, 1, cv::REDUCE_MAX);
    cv::reduce(tmp_ig_gnd, Imax_gnd, 1, cv::REDUCE_MAX);

    // Find indices where maximum value > 4
    std::vector<int> tmp_idx_hv, tmp_idx_gnd;
    for (int i = 0; i < Imax_hv.rows; ++i)
    {
        if (Imax_hv.at<uchar>(i,0) > 4)  
            tmp_idx_hv.push_back(i);
            
        if (Imax_gnd.at<uchar>(i,0) > 4) 
            tmp_idx_gnd.push_back(i);
    }

    cv::Mat blob = cv::Mat::ones(img.size(), CV_8U); 
    blob.convertTo(blob, CV_8U, 255);
    hv_blob_detection(img, tmp_idx_hv, particle_px_lim, blob);
    gnd_blob_detection(img, tmp_idx_gnd, particle_px_lim, blob);

    cv::bitwise_and(img, blob, img);

}


void Image_Processor::hv_blob_detection(const cv::Mat& img, 
                   std::vector<int>& tmp_idx, 
                   const std::vector<int>& particle_px_lim, 
                   cv::Mat& blob) {

    for (int i = 0; i < tmp_idx.size(); ++i) {
        // Finding index blob
        std::vector<int> tmp_idx_blob;
        for (int j = particle_px_lim[0]; j < particle_px_lim[1]; ++j) {
            if (img.at<uchar>(tmp_idx[i], j) > 4)
                tmp_idx_blob.push_back(j);
        }

        // Finding start and end of blob
        int blob_start, blob_end;
        if(!tmp_idx_blob.empty()) {
            blob_start = tmp_idx_blob[0];
            // Finding discontinuities
            int j = 1;
            for (; j < tmp_idx_blob.size(); ++j) {
                if ((tmp_idx_blob[j] - tmp_idx_blob[j-1]) > 4)
                    break;
            }
            blob_end = tmp_idx_blob[j-1] + 5;

            for (int k = blob_start; k <= blob_end; ++k) {
                blob.at<uchar>(tmp_idx[i], k) = 0;
            }
        }
    }
}


void Image_Processor::gnd_blob_detection(const cv::Mat& img, 
                   std::vector<int>& tmp_idx, 
                   const std::vector<int>& particle_px_lim, 
                   cv::Mat& blob) {


    for (int i = 0; i < tmp_idx.size(); ++i) {
        // Finding index blob
        std::vector<int> tmp_idx_blob;
        for (int j = particle_px_lim[0]; j < particle_px_lim[1]; ++j) {
            if (img.at<uchar>(tmp_idx[i], j) > 4)
                tmp_idx_blob.push_back(j);
        }

        if (tmp_idx_blob.empty() || tmp_idx_blob.back() < 120)
            continue;
        
        // Finding start and end of blob
        int blob_start, blob_end;
        blob_end = tmp_idx_blob.back();
        int j = tmp_idx_blob.size() - 1;
        for (; j > 0; --j) {
            if (tmp_idx_blob[j] - tmp_idx_blob[j-1] > 4) 
                break;    
        }
        blob_start = tmp_idx_blob[j] - 5;
        // Setting blob to zero
        for (int k = blob_start; k <= blob_end; ++k) {
            blob.at<uchar>(tmp_idx[i], k) = 0;
        }
    }
}


cv::Mat Image_Processor::particle_detection(cv::Mat& imgGray, std::vector<int>& thresh_arr, std::vector<int>& pix_lim) 
{
    cv::threshold(imgGray, imgGray, thresh_arr.back() - 1, 255, cv::THRESH_TOZERO);

    cv::Mat mask = cv::Mat::zeros(imgGray.size(), CV_8UC1); 
    mask.colRange(pix_lim[0] - 1, pix_lim[1] + 1).setTo(255); // Assuming imgGray is of type CV_8UC1. Column range is [40, 171] inclusive.
    cv::Mat imgGrayMasked;
    cv::bitwise_and(imgGray, mask, imgGrayMasked);

    //find connected area in processed image with seperated particles
    int numOfLables = cv::connectedComponentsWithStats(imgGrayMasked, labels, stats, centroids, 8, CV_32S);

    find_peak(imgGray, labels, stats, numOfLables, {pix_lim[0] - 1, pix_lim[1]});
    numOfLables = cv::connectedComponentsWithStats(imgGrayMasked, labels, stats, centroids, 8, CV_32S);


    std::vector<std::pair<double, double>> weighted_coordinates;
    for(size_t label_idx = 1; label_idx < numOfLables; ++label_idx) {
        const int left = stats.at<int>(label_idx, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label_idx, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label_idx, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label_idx, cv::CC_STAT_HEIGHT);
        double x_sum = 0;
        double y_sum = 0;
        double w_sum = 0;
        int max_intensity = -1;
        for (int y_coordinate = top; y_coordinate < top + height; ++y_coordinate) {
            for (int x_coordinate = left; x_coordinate < left + width; ++x_coordinate) {
                if (labels.at<int>(y_coordinate, x_coordinate) == label_idx) { // Access labels using `at<int>`
                    uchar intensity = imgGray.at<uchar>(y_coordinate, x_coordinate); // Access imgGray using `at<uchar>`
                    w_sum += intensity;
                    x_sum += intensity * (x_coordinate);
                    y_sum += intensity * (y_coordinate);
                }
            }
        }

        std::pair<double, double> weighted_pair(x_sum / w_sum, y_sum / w_sum);
        weighted_coordinates.push_back(weighted_pair);
    }
    
    cv::Mat result(weighted_coordinates.size(), 2, CV_64F);
    for(int i = 0; i < weighted_coordinates.size(); ++i) {
        result.at<double>(i, 0) = weighted_coordinates[i].first;
        result.at<double>(i, 1) = weighted_coordinates[i].second;
    }
    return result;


}

void Image_Processor::find_peak(cv::Mat& imgGray, cv::Mat& labels, cv::Mat& stats, int numOfLables, const std::pair<double, double>& col_range) {
    const int radius = 3;
    cv::Mat imgGray_new = imgGray.clone();
    
    int rows = imgGray.rows;

    for(size_t label_idx = 1; label_idx < numOfLables; ++label_idx) {
        const int left = stats.at<int>(label_idx, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label_idx, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label_idx, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label_idx, cv::CC_STAT_HEIGHT);

        for (int y_coordinate = top; y_coordinate < top + height; ++y_coordinate) {
            for (int x_coordinate = left; x_coordinate < left + width; ++x_coordinate) {
                int cur_pixel_value = imgGray_new.at<uchar>(y_coordinate, x_coordinate); 
                if(cur_pixel_value > 0) {
                    int positive_count = 0;
                    int negative_count = 0;
                    for(int k = y_coordinate - radius; k <= y_coordinate + radius; ++k) {
                        for(int h = x_coordinate - radius; h <= x_coordinate + radius; ++h) {
                            if(k < 0 or k >=rows or h < col_range.first or h > col_range.second)
                                continue;

                            int neig_value = imgGray_new.at<uchar>(k, h);
                            int diff = cur_pixel_value - neig_value;
                            if(diff < -15) {
                                negative_count++;
                            }
                            if(diff > -2) { 
                                positive_count++;
                            }
                        }
                    }
                    if((positive_count < 40 || negative_count > 3))
                        imgGray.at<uchar>(y_coordinate, x_coordinate) = 0; 

                }
            }
        }
    }

}