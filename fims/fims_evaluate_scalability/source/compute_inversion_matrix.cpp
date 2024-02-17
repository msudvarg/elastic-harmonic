#include "compute_inversion_matrix.h"
#include "math_helplers.h"
#include "paramt_FIMS.h"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>

Eigen::MatrixXd Compute_Inversion_Matrix::FIMS_inversion_matrix(double factor, Eigen::VectorXd& Zp_bnds, Eigen::VectorXd& Dp_bnds,
                                                                double T, double pressure, int chgs, double Veff, const std::string &charger_type, 
                                                                const std::string &inlet_ver, int case_number) {
    
    if(!is_initialised) {
        for (int chg = 1; chg <= chgs; ++chg) {
            std::string Zp_path = compose_path(case_number, chg, "Zp");
            Eigen::VectorXd Zp = load_vector_from_csv(Zp_path);
            Zp_vector.emplace_back(Zp);

            std::string Zp_n_path = compose_path(case_number, chg, "Zp_n");
            Eigen::VectorXd Zp_n = load_vector_from_csv(Zp_n_path);
            Zp_n_vector.emplace_back(Zp_n);

            std::string omega_path = compose_path(case_number, chg, "omega");
            Eigen::MatrixXd omega = load_matrix_from_csv(omega_path);
            omega_vector.emplace_back(omega);
        }
        is_initialised = true;
    }

    Eigen::VectorXd Zp_bnds_diff = Zp_bnds.head(Zp_bnds.size() - 1) - Zp_bnds.tail(Zp_bnds.size() - 1);
    if ((Zp_bnds_diff.array() < 0).any()) {
        throw std::runtime_error("Zp_bnds must be in descending order");
    }

    // Check if Dp_bnds is in ascending order
    Eigen::VectorXd Dp_bnds_diff = Dp_bnds.tail(Dp_bnds.size() - 1) - Dp_bnds.head(Dp_bnds.size() - 1);
    if ((Dp_bnds_diff.array() < 0).any()) {
        throw std::runtime_error("Dp_bnds must be in ascending order");
    }

    int Ni_Zp = 10;
    int Ni_Dp = 10;
    int len_Zp_bnds = Zp_bnds.size();
    int len_Dp_bnds = Dp_bnds.size();

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(len_Zp_bnds - 1, len_Dp_bnds - 1);
    Eigen::VectorXd dZp = Zp_bnds.tail(len_Zp_bnds - 1) - Zp_bnds.head(len_Zp_bnds - 1);
    Eigen::VectorXd dlnDp = (Dp_bnds.tail(len_Dp_bnds - 1).array().log()) - (Dp_bnds.head(len_Dp_bnds - 1).array().log());

    // calculate a new vector Zp_i which contains Ni_Zp equally spaced points between each adjacent pair of elements in the Zp_bnds vector.
    Eigen::VectorXd Zp_i(Ni_Zp * (len_Zp_bnds - 1));
    for (int i = 0; i < len_Zp_bnds - 1; ++i) {
        Eigen::VectorXd temp = Eigen::VectorXd::LinSpaced(Ni_Zp, 0, Ni_Zp - 1);
        temp = ((temp.array() + 0.5) * dZp[i] / Ni_Zp).matrix();
        Zp_i.segment(i * Ni_Zp, Ni_Zp) = Eigen::VectorXd::Constant(Ni_Zp, Zp_bnds[i]) + temp;
    }

    for (int chg = 1; chg <= chgs; ++chg) {
        Eigen::VectorXd& Zp = Zp_vector[chg - 1];
        Eigen::VectorXd& Zp_n = Zp_n_vector[chg - 1];
        Eigen::MatrixXd& omega = omega_vector[chg - 1];

        for (int i = 0; i < len_Dp_bnds - 1; ++i) {
            Eigen::VectorXd seq = Eigen::VectorXd::LinSpaced(Ni_Dp, 0, Ni_Dp - 1);
            seq = seq * 0.5 / Ni_Dp * dlnDp[i];
            Eigen::VectorXd Dp_i = (Eigen::VectorXd::Constant(Ni_Dp, log(Dp_bnds[i])) + seq).array().exp();
            Eigen::VectorXd Zp_Dp_i = Dps_to_Zps(Dp_i, T, pressure, chg); 
            Eigen::VectorXd D_i = Zp_Dp_i * (k * T) / (chg * e);
            Eigen::MatrixXd Gi = Eigen::MatrixXd::Zero(Zp_i.size(), Dp_i.size());
            Eigen::MatrixXd product = Eigen::VectorXd::Ones(Zp_i.size()) * charging_probability(Dp_i, T, chg, charger_type).cwiseProduct(Eigen::VectorXd::Ones(Dp_i.size())).cwiseProduct(FIMS_pen_eff(D_i)).cwiseQuotient(Zp_Dp_i).transpose();
            // Compute the interp_value matrix
            for (int k = 0; k < Zp_i.size(); ++k) {
                for (int j = 0; j < Dp_i.size(); ++j) {
                    double value = interp2(Zp_n, Zp, omega, Zp_i[k] / Zp_Dp_i[j], Zp_Dp_i[j]);
                    Gi(k, j) = value;
                }
            }

            Gi = Gi.cwiseProduct(product);
            Gi = (Gi.array().isNaN()).select(0, Gi);

            for (int j = 0; j < len_Zp_bnds - 1; ++j) {
                G(j, i) += Gi.block(j * Ni_Zp, 0, Ni_Zp, Dp_i.size()).sum() * factor * ((-dZp[j] / Ni_Zp) * (dlnDp[i] / Ni_Dp));
            }
        }
    }

    return G;
}


Eigen::VectorXd Compute_Inversion_Matrix::charging_probability(const Eigen::VectorXd& Dp, double T, int chg, const std::string& type) {

    Eigen::VectorXd charge_prob(Dp.size());
    Eigen::VectorXd Dp_nm = Dp * 1e9;

    if (type == "RAD") {
        switch (chg) {
            case 1: {
                std::vector<double> a = {0.032014741,	-0.154428052, -0.001286347, 0.479995684, 0.604427911, -2.348356119};
                charge_prob = polyval_vector(a, Dp_nm.array().log10());
                charge_prob = (Eigen::ArrayXd::Constant(charge_prob.size(), 10.0).pow(charge_prob.array())).matrix();
                break;
            }
            case 2: {
                std::vector<double> a = {0.504332912, -5.741778918, 26.42343574, -62.83747051, 79.32475073, -44.45511997};
                charge_prob = polyval_vector(a, Dp_nm.array().log10());
                charge_prob = (Eigen::ArrayXd::Constant(charge_prob.size(), 10.0).pow(charge_prob.array())).matrix();
                charge_prob = (Dp_nm.array() < 20).select(0.0, charge_prob);
                break;
            }
            case -1: {
                std::vector<double> a = {0.0296988733351844, -0.126033901087282, -0.110547206662251, 0.620082691427975, 0.617535986184619, -2.31971809364052};
                charge_prob = polyval_vector(a, Dp_nm.array().log10());
                charge_prob = (Eigen::ArrayXd::Constant(charge_prob.size(), 10.0).pow(charge_prob.array())).matrix();
                break;
            }
            case -2: {
                std::vector<double> a = {0.10494942602229, -1.30692296482325, 7.07908214954542, -21.4452185911435, 35.888874766558, -26.3267066662961};
                charge_prob = polyval_vector(a, Dp_nm.array().log10());
                charge_prob = (Eigen::ArrayXd::Constant(charge_prob.size(), 10.0).pow(charge_prob.array())).matrix();
                charge_prob = (Dp_nm.array() < 20).select(0.0, charge_prob);
                break;
            }
        }
    }
    //other charging methods to be added
    return charge_prob;
}

Eigen::VectorXd Compute_Inversion_Matrix::polyval_vector(const std::vector<double>& coeffs, const Eigen::ArrayXd& x) {
    Eigen::VectorXd result(x.size());
    result.setZero();

    for (size_t i = 0; i < coeffs.size(); ++i) {
        result += coeffs[i] * x.pow(double(coeffs.size() - i - 1)).matrix();
    }
    return result;
}


Eigen::VectorXd Compute_Inversion_Matrix::FIMS_pen_eff(const Eigen::VectorXd& D) {
    auto& fims = paramt_FIMS::getInstance();
    
    Eigen::VectorXd eff;

    if (fims.inlet_ver == "FIMS_V2.3") {
        auto eff_diff_1 = particle_pen_eff_diffusion(D, std::vector<double>(fims.Q.begin(), fims.Q.end() - 1), fims.L);
        auto eff_diff_2 = particle_pen_eff_diffusion(D, {fims.Q.back(), fims.Q.back() * 0.5, fims.Q.back() * 0.5}, {1.25 * 0.0254, 1.25 * 0.0254, 2.21 * 0.0254});
        auto eff_channel = particle_pen_eff_channel(D, {fims.Q.back(), fims.Q.back()}, {1.1 * 0.0254, 0.96 * 0.0254}, {fims.inlet_eff_size[0] / 5, fims.inlet_eff_size[1] / 5});

        eff = eff_diff_1.array() * eff_diff_2.array() * eff_channel.array();
    } else {
        throw std::runtime_error("Unsupported inlet case. Only \"FIMS_V2.3\" is supported now.");
    }
    return eff;
}


Eigen::VectorXd Compute_Inversion_Matrix::particle_pen_eff_channel(const Eigen::VectorXd& D, const std::vector<double>& Q, const std::vector<double>& L, 
                                        const std::vector<double>& gap2width, int option) {
    Eigen::VectorXd eff = Eigen::VectorXd::Ones(D.size());

    switch (option) {
        case 1:
            for (int i = 0; i < Q.size(); ++i) {
                if (i == 0) {
                    eff = (-7.868 * L[i] * D.array() / (Q[i] / 60000 * gap2width[i])).exp();
                } else {
                    eff = eff.array() * (-7.868 * L[i] * D.array() / (Q[i] / 60000 * gap2width[i])).exp();
                }
            }
            break;

        case 2:
            for (int i = 0; i < Q.size(); ++i) {
                Eigen::ArrayXd mu = L[i] * D.array() / (Q[i] / 60000 * gap2width[i]);
                Eigen::Array<bool, Eigen::Dynamic, 1> tmp_idx = mu < 0.005;

                Eigen::VectorXd eff_update_true(D.size());
                eff_update_true.array() = (1 - 2.96 * mu.pow(2.0 / 3.0) + 0.4 * mu).matrix().array();

                Eigen::VectorXd eff_update_false(D.size());
                eff_update_false.array() = (0.91 * (-7.54 * mu).exp() + 0.0531 * (-85.7 * mu).exp()).matrix().array();

                eff_update_true = tmp_idx.select(eff_update_true, eff_update_false);
                if (i == 0) {
                    eff = eff_update_true;
                } else {
                    eff = eff.array() * eff_update_true.array();
                }
            }
            break;
    }

    return eff;
}


Eigen::VectorXd Compute_Inversion_Matrix::particle_pen_eff_diffusion(const Eigen::VectorXd& D, const std::vector<double>& Q, const std::vector<double>& L) {
    Eigen::VectorXd eff = Eigen::VectorXd::Ones(D.size());

    for (int i = 0; i < Q.size(); ++i) {
        if (i == 0) {
            eff = (-5.5 * (L[i] * D.array() / (Q[i] / 60000)).pow(2.0 / 3.0)).exp();
        } else {
            eff = eff.array() * (-5.5 * (L[i] * D.array() / (Q[i] / 60000)).pow(2.0 / 3.0)).exp();
        }
    }

    return eff;
}


double Compute_Inversion_Matrix::Zp_to_Dp(double Zp, double T, double pressure, int chg) {
    double viscosity = 1.8334e-5 * std::pow(T / 296.15, 1.5) * (296.15 + 110.4) / (T + 110.4);
    double mfp = mfp_1atm_298K * (T / 296.15) / (pressure / 1.0) * (1 + 110.4 / 296.15) / (1 + 110.4 / T);

    if (std::isnan(Zp)) {
        Zp = 1e-6;
    }

    double Dp_high = 6.628 * mfp / (std::sqrt(1 + 4 * 3.314 * mfp * (3 * M_PI * viscosity * Zp / (e * chg))) - 1);
    double Dp_low = chg * e / (3.0 * M_PI * viscosity * Zp);
    double Dp_Cc = chg * e / (3.0 * M_PI * viscosity * Zp);
    double Dp_mid;

    while (true) {
        Dp_mid = (Dp_high + Dp_low) / 2.0;
        double Cc = 1.0 + (2.0 * mfp / Dp_mid) * (1.257 + 0.4 * std::exp(-1.1 * Dp_mid / (2.0 * mfp)));

        if (std::abs(((Dp_mid / Cc) - Dp_Cc) / Dp_Cc) < 0.001) {
            break;
        } else if ((Dp_mid / Cc) > Dp_Cc) {
            Dp_high = Dp_mid;
        } else {
            Dp_low = Dp_mid;
        }
    }

    return Dp_mid;
}


Eigen::VectorXd Compute_Inversion_Matrix::Zps_to_Dps(const Eigen::VectorXd& Zp, double T, double pressure, int chg) {

    double viscosity = 1.8334e-5 * std::pow(T / 296.15, 1.5) * (296.15 + 110.4) / (T + 110.4);
    double mfp = mfp_1atm_298K * (T / 296.15) / (pressure / 1.0) * (1 + 110.4 / 296.15) / (1 + 110.4 / T);

    Eigen::VectorXd Dp_high = 6.628 * mfp / ((Eigen::VectorXd::Ones(Zp.size()).array() + 4 * 3.314 * mfp * (3 * M_PI * viscosity * Zp.array() / (e * chg))).sqrt() - 1).array();
    Eigen::VectorXd Dp_low = chg * e / (3.0 * M_PI * viscosity * Zp.array());

    Eigen::VectorXd Dp_Cc = chg * e / (3.0 * M_PI * viscosity * Zp.array());
    Eigen::VectorXd Dp = Eigen::VectorXd::Zero(Zp.size());

    double tolerance = 0.001;

    for (int i = 0; i < Zp.size(); ++i) {
        while (true) {
            double Dp_mid = (Dp_high[i] + Dp_low[i]) / 2.0;
            double Cc = 1.0 + (2.0 * mfp / Dp_mid) * (1.257 + 0.4 * std::exp(-1.1 * Dp_mid / (2.0 * mfp)));

            if (std::abs(((Dp_mid / Cc) - Dp_Cc[i]) / Dp_Cc[i]) < tolerance) {
                Dp[i] = Dp_mid;
                break;
            } else if ((Dp_mid / Cc) > Dp_Cc[i]) {
                Dp_high[i] = Dp_mid;
            } else {
                Dp_low[i] = Dp_mid;
            }
        }
    }

    return Dp;
}


Eigen::VectorXd Compute_Inversion_Matrix::Dps_to_Zps(const Eigen::VectorXd& Dp, double T, double pressure, int chg) {

    double viscosity = 1.8334e-5 * std::pow(T / 296.15, 1.5) * (296.15 + 110.4) / (T + 110.4);
    double mfp = mfp_1atm_298K * (T / 296.15) / (pressure / 1.0) * (1 + 110.4 / 296.15) / (1 + 110.4 / T);

    Eigen::VectorXd Cc = 1.0 + (2.0 * mfp * Eigen::VectorXd::Ones(Dp.size())).array().cwiseQuotient(Dp.array()).cwiseProduct(1.257 + 0.4 * (-1.1 * Dp.array() / (2.0 * mfp)).exp());
    Eigen::VectorXd Zp = chg * e * Cc.array().cwiseQuotient(3.0 * M_PI * viscosity * Dp.array());   

    return Zp;
}


std::string Compute_Inversion_Matrix::compose_path(int case_number, int chg_number, const std::string type) {
    std::stringstream path_ss;
    if (type == "omega" or type == "Zp" or type == "Zp_n") {
        path_ss << "../transfer_function/case_" << case_number << "/combined_transfer_function/" << type << chg_number << ".csv";
    }
    else {
        std::runtime_error("Error: path does not exist!");
    }

    std::string path = path_ss.str();
    return path;
}

Eigen::MatrixXd Compute_Inversion_Matrix::load_matrix_from_csv(const std::string& file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_name << std::endl;
        return Eigen::MatrixXd();
    }

    std::vector<double> values;
    std::string line;
    int rows = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;

        while (std::getline(iss, token, ',')) {
            try {
                double value = std::stod(token);
                values.push_back(value);
            } catch (const std::invalid_argument&) {
                std::cerr << "Warning: Unable to parse value '" << token << "' as a double in " << file_name << ". Assigning NaN." << std::endl;
                values.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }

        rows++;
    }

    if (rows == 0) {
        std::cerr << "Error: Empty or invalid CSV file." << std::endl;
        return Eigen::MatrixXd();
    }

    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);

}

Eigen::VectorXd Compute_Inversion_Matrix::load_vector_from_csv(const std::string& file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_name << std::endl;
        return Eigen::VectorXd();
    }

    std::vector<double> values;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;

        while (std::getline(iss, token, ',')) {
            try {
                double value = std::stod(token);
                values.push_back(value);
            } catch (const std::invalid_argument&) {
                std::cerr << "Warning: Unable to parse value '" << token << "' as a double in " << file_name << ". Assigning NaN." << std::endl;
                values.push_back(std::nan(""));
            }
        }
    }

    if (values.empty()) {
        std::cerr << "Error: Empty or invalid CSV file." << std::endl;
        return Eigen::VectorXd();
    }

    return Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
}



double Compute_Inversion_Matrix::interp2(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const Eigen::MatrixXd& z,
               double xq, double yq) {


    int ix = -1;
    int iy = -1;

    if (z.rows() != y.size() and z.cols() != x.size()) {
        throw std::runtime_error("Dimensions of z do not match sizes of x and y.");
    }

    auto x_upper = std::upper_bound(x.data(), x.data() + x.size(), xq);
    ix = x_upper - x.data() - 1;

    auto y_lower = std::find_if(y.data(), y.data() + y.size(), [&](double val) { return val <= yq; });
    iy = y_lower - y.data() - 1;

    if (ix < 0 or ix >= x.size() - 1 or iy < 0 or iy >= y.size() - 1) {
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








