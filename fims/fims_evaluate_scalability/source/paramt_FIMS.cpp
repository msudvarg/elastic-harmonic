#include "paramt_FIMS.h"

#include <fstream>

paramt_FIMS::paramt_FIMS(){
    get_Dp_coordinates();
}

void paramt_FIMS::get_Dp_coordinates() {
    dlnDp_fixed = log(Dp_max / Dp_min) / (N - 1);
    Eigen::VectorXd range = Eigen::VectorXd::LinSpaced(N, 0, N - 1);
    Dp_fixed = (log(Dp_min) + range.array() * dlnDp_fixed).exp();

    int len = Dp_fixed.size();
    Dp_fixed_set.resize(3, len);
    Dp_fixed_set.row(1) = Dp_fixed;

    // Calculate the first row of Dp_fixed_set
    Dp_fixed_set(0, 0) = std::exp(std::log(Dp_fixed(0)) - 0.5 * (std::log(Dp_fixed(1)) - std::log(Dp_fixed(0))));
    for (int i = 1; i < len; ++i) {
        Dp_fixed_set(0, i) = std::exp(0.5 * (std::log(Dp_fixed(i-1)) + std::log(Dp_fixed(i))));
    }

    // Calculate the third row of Dp_fixed_set
    Dp_fixed_set.block(2, 0, 1, len - 1) = Dp_fixed_set.block(0, 1, 1, len - 1);
    Dp_fixed_set(2, len - 1) = std::exp(std::log(Dp_fixed(len - 1)) + 0.5 * (std::log(Dp_fixed(len - 1)) - std::log(Dp_fixed(len - 2))));

}

void paramt_FIMS::load_config(const std::string& path) {
    try {
        std::cout << "path: " << path << std::endl;
        cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        // Check if the file was opened successfully
        if (!fs.isOpened()) {
            throw std::runtime_error("Could not open config file: " + path);
        }
        config_paramt(fs);

    } catch (cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }
}

void paramt_FIMS::config_paramt(cv::FileStorage& fs) {
    //dimension, flow rates and pressure
    flight_num = (int)fs["flight_num"];
    Npx = (int)fs["Npx"];
    Npy = (int)fs["Npy"];
    spot2pyc = (double)fs["spot2pyc"];

    a = (double)fs["a"];
    b = (double)fs["b"];
    Ls = (double)fs["Ls"];
    Lc = (double)fs["Lc"];

    x_min = a * (double)fs["x_min"];
    x_max = a * (double)fs["x_max"];
    y_min = (double)fs["y_min"];
    y_max = (double)fs["y_max"];
    
    Qt_set = (double)fs["Qt_set"];
    P_ratio = (double)fs["P_ratio"];
   
    //data inversion
    housekeeping_ver = (double)fs["housekeeping_ver"];
    charger_type = (std::string)fs["charger_type"];
    HV_level_method = (int)fs["HV_level_method"];
    Qa_min_for_inv = (double)fs["Qa_min_for_inv"];
    bnd_interval = (double)fs["bnd_interval"];
    max_resident_time = (double)fs["max_resident_time"];

    cv::FileNode Vm_rangeNode = fs["Vm_range"];
    std::vector<int> Vm_range_;
    for (cv::FileNodeIterator it = Vm_rangeNode.begin(); it != Vm_rangeNode.end(); ++it) {
        Vm_range_.push_back((int)(*it));
    }
    Vm_range = Vm_range_;
    Veff = Vm_range;

    cv::FileNode case_numNode = fs["case_num"];
    std::vector<int> case_num_;
    for (cv::FileNodeIterator it = case_numNode.begin(); it != case_numNode.end(); ++it) {
        case_num_.push_back((int)(*it));
    }
    case_num = case_num_;

    dt = (double)fs["dt"];
    N_Dp_bin = (int)fs["N_Dp_bin"];
    chgs = (double)fs["chgs"];

    Dp_min = (double)fs["Dp_min"];
    Dp_max = (double)fs["Dp_max"];

    N = (int)fs["N"];
    
    //inlet parameters
    inlet_ver = (std::string)fs["inlet_ver"];
    cv::FileNode inlet_eff_sizeNode = fs["inlet_eff_size"];
    std::vector<double> inlet_eff_size_;
    for (cv::FileNodeIterator it = inlet_eff_sizeNode.begin(); it != inlet_eff_sizeNode.end(); ++it) {
        inlet_eff_size_.push_back((double)(*it));
    }
    inlet_eff_size = inlet_eff_size_;
    
    cv::FileNode QNode = fs["Q"];
    std::vector<double> Q_;
    for (cv::FileNodeIterator it = QNode.begin(); it != QNode.end(); ++it) {
        Q_.push_back((double)(*it));
    }
    Q = Q_;

    cv::FileNode LNode = fs["L"];
    std::vector<double> L_;
    for (cv::FileNodeIterator it = LNode.begin(); it != LNode.end(); ++it) {
        L_.push_back((double)(*it));
    }
    L = L_;

    cv::FileNode Q_pre_FIMSNode = fs["Q_pre_FIMS"];
    std::vector<double> Q_pre_FIMS_;
    for (cv::FileNodeIterator it = Q_pre_FIMSNode.begin(); it != Q_pre_FIMSNode.end(); ++it) {
        Q_pre_FIMS_.push_back((double)(*it));
    }
    Q_pre_FIMS = Q_pre_FIMS_;

    cv::FileNode L_pre_FIMSNode = fs["L_pre_FIMS"];
    std::vector<double> L_pre_FIMS_;
    for (cv::FileNodeIterator it = L_pre_FIMSNode.begin(); it != L_pre_FIMSNode.end(); ++it) {
        L_pre_FIMS_.push_back((double)(*it));
    }
    L_pre_FIMS = L_pre_FIMS_;
    
    Q.insert(Q.begin(), Q_pre_FIMS.begin(), Q_pre_FIMS.end());
    L.insert(L.begin(), L_pre_FIMS.begin(), L_pre_FIMS.end());

    cv::FileNode hv_px0Node = fs["hv_px0"];
    std::vector<double> hv_px0_;
    for (cv::FileNodeIterator it = hv_px0Node.begin(); it != hv_px0Node.end(); ++it) {
        hv_px0_.push_back((double)(*it));
    }
    hv_px0 = std::make_pair(hv_px0_[0], hv_px0_[1]);

    cv::FileNode gnd_px0Node = fs["gnd_px0"];
    std::vector<double> gnd_px0_;
    for (cv::FileNodeIterator it = gnd_px0Node.begin(); it != gnd_px0Node.end(); ++it) {
        gnd_px0_.push_back((double)(*it));
    }
    gnd_px0 = std::make_pair(gnd_px0_[0], gnd_px0_[1]);

    cv::FileNode hv_slopeNode = fs["hv_slope"];
    std::vector<double> hv_slope_;
    for (cv::FileNodeIterator it = hv_slopeNode.begin(); it != hv_slopeNode.end(); ++it) {
        hv_slope_.push_back((double)(*it));
    }
    hv_slope = std::make_pair(hv_slope_[0], hv_slope_[1]);

    cv::FileNode gnd_slopeNode = fs["gnd_slope"];
    std::vector<double> gnd_slope_;
    for (cv::FileNodeIterator it = gnd_slopeNode.begin(); it != gnd_slopeNode.end(); ++it) {
        gnd_slope_.push_back((double)(*it));
    }
    gnd_slope = std::make_pair(gnd_slope_[0], gnd_slope_[1]);

    cv::FileNode pycNode = fs["pyc"];
    std::vector<double> pyc_;
    for (cv::FileNodeIterator it = pycNode.begin(); it != pycNode.end(); ++it) {
        pyc_.push_back((double)(*it));
    }
    pyc = std::make_pair(pyc_[0], pyc_[1]);

    // To convert a list in YAML to std::vector in C++
    cv::FileNode thresh_arrNode = fs["thresh_arr"];
    std::vector<int> thresh_arr_;
    for (cv::FileNodeIterator it = thresh_arrNode.begin(); it != thresh_arrNode.end(); ++it) {
        thresh_arr_.push_back((int)(*it));
    }
    thresh_arr = thresh_arr_;
    
    cv::FileNode area_arrNode = fs["area_arr"];
    std::vector<int> area_arr_;
    for (cv::FileNodeIterator it = area_arrNode.begin(); it != area_arrNode.end(); ++it) {
        area_arr_.push_back((int)(*it));
    }
    area_arr = area_arr_;

    cv::FileNode pix_limNode = fs["pix_lim"];
    std::vector<int> pix_lim_;
    for (cv::FileNodeIterator it = pix_limNode.begin(); it != pix_limNode.end(); ++it) {
        pix_lim_.push_back((int)(*it));
    }
    pix_lim = pix_lim_;

    cv::FileNode pxlim_gndNode = fs["pxlim_gnd"];
    std::vector<int> pxlim_gnd_;
    for (cv::FileNodeIterator it = pxlim_gndNode.begin(); it != pxlim_gndNode.end(); ++it) {
        pxlim_gnd_.push_back((double)(*it));
    }
    pxlim_gnd = pxlim_gnd_;

    cv::FileNode pxlim_hvNode = fs["pxlim_hv"];
    std::vector<int> pxlim_hv_;
    for (cv::FileNodeIterator it = pxlim_hvNode.begin(); it != pxlim_hvNode.end(); ++it) {
        pxlim_hv_.push_back((double)(*it));
    }
    pxlim_hv = pxlim_hv_;

    get_Dp_coordinates();

    HK_reading_duration = (double)fs["HK_reading_duration"];
    image_processing_duration = (double)fs["image_processing_duration"];
    data_inversion_duration = (double)fs["data_inversion_duration"];
}



