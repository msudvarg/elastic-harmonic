#pragma once
#include <iostream>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <Eigen/Dense>



class paramt_FIMS {
public:
/*************image processing parameters**************/
    std::vector<int> thresh_arr = {100, 80, 60, 40, 30, 20, 15, 10, 8, 6};
    std::vector<int> area_arr = {1,1,1,1,1,1,1,1,1,1};
    std::vector<int> pix_lim = {39, 170};
    std::vector<int> pxlim_gnd = {170, 185};
    std::vector<int> pxlim_hv = {21, 35};        

    std::pair<double, double> hv_px0 = {29.94, 24.73}; 
    std::pair<double, double> gnd_px0 = {182.60, 176.3};
    std::pair<double, double> hv_slope = {7.0e-4, 5.0e-4};
    std::pair<double, double> gnd_slope = {1.0e-4, -1.0e-4};
    std::pair<double, double> pyc = {532.28, 529.28};

/*************FIMS dimension, flow rates, and pressure**************/
    /* physical dimention of fims instrument */
    int flight_num = 1;

    int Npx = 196;    // # of pxl in x-direction
    int Npy = 1024;   // # of pxl in y-direction
    double spot2pyc = -15.22; // 524-539.22 Constant value

    double a = 11.176e-3; // Gap width (x-direction)
    double b = 12.7e-2; // Electrode width (y-direction)
    double Ls = 174.3202e-3; // Effective length of electrode
    double Lc = 0.365277399999999; // Condenser length (2.659+4.675-6.863+1.34+0.5+0.32+11.5+0.25)*0.0254

    /* view window */
    double x_min = a * 0.2;
    double x_max = a * 0.8;
    double y_min = -0.035;
    double y_max = 0.035;

    double Qt_set = 1.056*(13+0.26)*1e-3/60; //Effective total flowrate
    double P_ratio=0.95;	//Adjusted to typical value observed at Wallops pre-CAMP2Ex


/**************** Inversion parameters ********************/
    int housekeeping_ver = 3; // FIMS version deployed during CAMP2Ex Pabs corrected directly in LabView
    std::string charger_type = "RAD"; // Using Po-210 charger during CAMP2Ex Set to 'SPX' for x-ray charger
    int HV_level_method = 2;
    double Qa_min_for_inv=0.05;

    double bnd_interval = 1; //interval for time boundaries
    double max_resident_time = 3;
    std::vector<int> Vm_range = {6000,5000,4000,3000,2000};
    std::vector<int> Veff = Vm_range;
    std::vector<int> case_num = {1361,1551,3041,1331,1121};
    int dt = 1;   //average FIMS data into X second intervals. 1 is ideal, may need to be higher if concentrations low. ACEENA used 5.
    int N_Dp_bin = 30; //number of Dp bins
    int chgs = 2; //Inv. matrix is calculated up to this #chgs/particles

    double Dp_min = 10e-9;
    double Dp_max = 600e-9;
    int N = 30;
    double dlnDp_fixed;
    Eigen::VectorXd Dp_fixed;
    Eigen::MatrixXd Dp_fixed_set;
    bool use_normalized_inversion = false;
    double minI_for_inversion = 6;


    /************************** inlet parameters **********************/
    std::string inlet_ver = "FIMS_V2.3";
    std::vector<double> inlet_eff_size = {0.04,0.0095} ;

    std::vector<double> Q_pre_FIMS = {8.89,8.89,3.89,3.89,3.89,3.89,1.97,1.01,1.01};
    std::vector<double> L_pre_FIMS = {5.00,0.72,0.12,0.10,0.30,0.10,0.42,0.09,0.04};

    std::vector<double> Q = {8.89, 8.89, 3.89, 3.89, 3.89, 3.89, 1.97, 1.01, 1.01, 0.26, 0.26};
    std::vector<double> L = {5.00, 0.72, 0.12, 0.10, 0.30, 0.10, 0.42, 0.09, 0.04, 4.5534};

    static paramt_FIMS& getInstance(const std::string& configPath = "") {
        static paramt_FIMS instance; 
        if (!configPath.empty()) {
            instance.load_config(configPath);
        }
        return instance;
    }

    /***************************config duration**************************************/
    double HK_reading_duration = 500;
    double image_processing_duration = 100;
    double data_inversion_duration = 1000;

    std::unordered_map<int, std::string> test_images = {
        {100, "test_images"},
        {200, "2stack"},
        {300, "3stack"},
        {400, "4stack"},
        {500, "5stack"},
        {600, "6stack"},
        {700, "7stack"},
        {800, "8stack"},
        {900, "9stack"},
        {1000, "10stack"},
    };

    
private:

    paramt_FIMS();
    void get_Dp_coordinates();

    // Deleted copy constructor, assignment operator and move syntax
    paramt_FIMS(const paramt_FIMS&) = delete;
    paramt_FIMS(paramt_FIMS&&) = delete;
    paramt_FIMS& operator=(const paramt_FIMS&) = delete;
    paramt_FIMS& operator=(paramt_FIMS&&) = delete;

    void load_config(const std::string& path);
    void config_paramt(cv::FileStorage& fs);
    
};