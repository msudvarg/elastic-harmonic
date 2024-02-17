/*

fims_get_periods.cc

This produces periods for the last table of
Section VI.C: "Evaluating Scalability" of FIMS

Run with ./fims_get_periods c_image e_image c_hk e_hk c_inversion e_inversion
Generates periods for utilization bounds of 0.1--0.5

*/

#include "harmonic.h"

#include <iostream>
#include <array>
#include <cmath>

Harmonic_Elastic elastic_space {3};

int main(int argc, char * argv[]) {

    //Task takes T_min, T_max, C, E
    elastic_space.add_task(Task {100, 1000, atof(argv[1]), atof(argv[2])}); //Image Processing
    elastic_space.add_task(Task {500, 5000, atof(argv[3]), atof(argv[4])}); //HK Data Reading
    elastic_space.add_task(Task {1000, 10000, atof(argv[5]), atof(argv[6])}); //Data Inversion

    if(!elastic_space.generate()) {
        std::cout << "Could not make harmonic assignment!" << std::endl;
        return 0;
    }

    std::array<float, 5> utilizations {0.5, 0.4, 0.3, 0.2, 0.1};

    std::cout << "Utilization T_image T_hk T_inv" << std::endl;

    for (float u : utilizations) {        
        Chain * chain = elastic_space.assign_periods_slow(u);
        if(!chain) {
            std::cout << "Utilization " << u << " is too low!" << std::endl;
            return 0;
        }

        const Tasks tasks = elastic_space.get_tasks();

        std::cout << u << ' '
                  << ceil(tasks[0].t) << ' '
                  << ceil(tasks[1].t) << ' '
                  << ceil(tasks[2].t) << std::endl;
    }


    return 0;
}