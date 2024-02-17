#pragma once

#include <Eigen/Dense>

class Compute_Inversion_Matrix {
public:
    static Compute_Inversion_Matrix& getInstance() {
        static Compute_Inversion_Matrix instance;
        return instance;
    }

    Eigen::MatrixXd FIMS_inversion_matrix(double factor, Eigen::VectorXd& Zp_bnds, Eigen::VectorXd& Dp_bnds,
                                        double T, double pressure, int chgs, double Veff, const std::string &charger_type, 
                                        const std::string &inlet_ver, int case_number);
    
    double Zp_to_Dp(double Zp, double T, double pressure, int chg);
    Eigen::VectorXd Zps_to_Dps(const Eigen::VectorXd& Zp, double T, double pressure, int chg);
    Eigen::VectorXd Dps_to_Zps(const Eigen::VectorXd& Dp, double T, double pressure, int chg);

    Eigen::MatrixXd load_matrix_from_csv(const std::string& file_name);
    Eigen::VectorXd load_vector_from_csv(const std::string& file_name);

    std::string compose_path(int case_number, int chg_number, std::string type);

private:

    const double k = 1.381e-23;
    const double mfp_1atm_298K = 65.1e-9;
    const double e = 1.602e-19;
    const double const_ep0 = 8.854e-12;

    // const double mfp_1atm_298K = 6.62e-8;
    // const double e = 1.602176634e-19;
    // const double k = 1.380649e-23;
    // const double const_ep0 = 8.8541878128e-12;
    const double pi = 3.14159265358979323846;

    double ratio_of_pos2neg_ion_conc = 1.0;
    double ratio_of_pos2neg_ion_mob = 0.875;

    bool is_initialised = false;
    std::vector<Eigen::VectorXd> Zp_vector;
    std::vector<Eigen::VectorXd> Zp_n_vector;
    std::vector<Eigen::MatrixXd> omega_vector;



    Eigen::VectorXd charging_probability(const Eigen::VectorXd& Dp, double T, int chg, const std::string& type);
    Eigen::VectorXd polyval_vector(const std::vector<double>& coeffs, const Eigen::ArrayXd& x);
    // Eigen::VectorXd FIMS_act_eff(const Eigen::VectorXd& Dp_nm, double Veff);
    Eigen::VectorXd FIMS_pen_eff(const Eigen::VectorXd& D);

    Eigen::VectorXd particle_pen_eff_channel(const Eigen::VectorXd& D, const std::vector<double>& Q, const std::vector<double>& L, 
                                        const std::vector<double>& gap2width, int option = 1);

    Eigen::VectorXd particle_pen_eff_diffusion(const Eigen::VectorXd& D, const std::vector<double>& Q, const std::vector<double>& L);
    double interp2(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::MatrixXd& z,
               double xq, double yq);

    Compute_Inversion_Matrix() = default;
    Compute_Inversion_Matrix(const Compute_Inversion_Matrix&) = delete;
    Compute_Inversion_Matrix& operator=(const Compute_Inversion_Matrix&) = delete;

};