#include <string>
#include <mutex>
#include <deque>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <atomic>
#include <condition_variable>

class Image_Processor {
public:
    Image_Processor(int flight_num, std::mutex& data_mutex, std::condition_variable& data_cv, 
                    std::deque<std::pair<double, std::vector<cv::Point2d>>>& particle_deque, std::atomic<double>& frame_time);
    
    void start();
    void stop();

private:
    std::string filename_;
    int flight_num_;
    std::mutex& data_mutex_;
    std::condition_variable& data_cv_;
    std::deque<std::pair<double, std::vector<cv::Point2d>>>& particle_deque_;
    std::atomic<double>& frame_time_;
    std::thread processor_thread_;
    bool stop_thread_;
    double start_time;
    double next_bin_time;

    cv::Mat labels;
    cv::Mat stats; 
    cv::Mat centroids;
    // cv::Mat BW;

    // void zmqread_n_process();
    void fileread_n_process();
    void shmread_n_process();

    double convert_time2hours (std::string & timestamp);
    void get_files(const std::string& dir, std::vector<std::string>& files);
    void image_process(cv::Mat& img, double t);
    //blob removal functions

    cv::Mat particle_detection_par1(cv::Mat& ig, std::vector<int>& thresh_arr, std::vector<int>& area_arr, std::vector<int>& pix_lim);
    
    //particles detection functions
    cv::Mat particle_detection_par(cv::Mat& ig, std::vector<int>& thresh_arr, std::vector<int>& area_arr, std::vector<int>& pix_lim);
    void remove_double_counting_par(std::vector<std::vector<std::pair<double, double>>>& this_frame_coordinates,
                            std::vector<std::pair<double, double>>& this_frame_coordinates_final,
                            std::vector<int>& this_frame_maxvals_final,
                            std::vector<int>& this_frame_areas_final,
                            std::vector<std::vector<int>>& this_frame_maxvals,
                            std::vector<std::vector<int>>& this_frame_areas,
                            std::vector<int>& pix_lim);
    void compute_weighted_centroids_par(int numObj, cv::Mat& centroids, cv::Mat& stats, cv::Mat& labels, cv::Mat& ig,
                                std::vector<std::pair<double, double>>& coordinates,
                                std::vector<std::pair<double, double>>& weighted_coordinates,
                                std::vector<int>& areas,
                                std::vector<int>& max_val);


    cv::Mat particle_detection(cv::Mat& ig, std::vector<int>& thresh_arr, std::vector<int>& area_arr, std::vector<int>& pix_lim);

    cv::Mat particle_detection(cv::Mat& imgGray, std::vector<int>& thresh_arr, std::vector<int>& pix_lim);
    void find_peak(cv::Mat& imgGray, cv::Mat& labels, cv::Mat& stats, int numOfLables, const std::pair<double, double>& col_range);

    void remove_double_counting(std::vector<std::pair<double, double>>& this_frame_coordinates,
                            std::vector<std::pair<double, double>>& this_frame_coordinates_final,
                            std::vector<int>& this_frame_maxvals_final,
                            std::vector<int>& this_frame_areas_final,
                            std::vector<int>& this_frame_maxvals,
                            std::vector<int>& this_frame_areas,
                            std::vector<int>& pix_lim);
    void compute_weighted_centroids(int numObj, cv::Mat& centroids, cv::Mat& stats, cv::Mat& labels, cv::Mat& ig,
                                std::vector<std::pair<double, double>>& coordinates,
                                std::vector<std::pair<double, double>>& weighted_coordinates,
                                std::vector<int>& areas,
                                std::vector<int>& max_val);

    //wall detection functions
    cv::Vec4d find_wall(cv::Mat ig, int offset);
    cv::Vec4d channel_wall_positions_from_image(cv::Mat& ig, std::vector<int>& pxlim_hv, std::vector<int>& pxlim_gnd);

    //coordinate converting functions
    cv::Mat convert_coordinates (cv::Mat& particles, cv::Vec4d& wall, std::pair<double, double>& hv_px0, std::pair<double, double>& gnd_px0, 
        const std::pair<double, double>& hv_slope, const std::pair<double, double>& gnd_slope, const std::pair<double, double>& pixel_yc, double a);
    cv::Mat convert_coordinates2 (cv::Mat& particles, cv::Vec4d& hv_wall, double wall_distance,  double scale);
    cv::Mat filter_particles (const cv::Mat& particles, double xmin, double xmax, double ymin, double ymax);
    
    //sort particle functions
    cv::Mat correct_particle_time(double t, const cv::Mat& particles, double Qt, double a, double b, double Lc);
    void put_to_bin(cv::Mat& particles, cv::Mat& time, double interval);
    void sort_particles (cv::Mat& particles, double t, double Qt, double a, 
                                        double b, double Lc, double interval);


    void blob_removal(cv::Mat& img, std::vector<int>& particle_px_lim);
    void hv_blob_detection(const cv::Mat& img, std::vector<int>& tmp_idx, const std::vector<int>& particle_px_lim, cv::Mat& blob);
    void gnd_blob_detection(const cv::Mat& img, std::vector<int>& tmp_idx, const std::vector<int>& particle_px_lim, cv::Mat& blob);   

};

