#include "harmonic.h"

#include <iostream>

Harmonic_Elastic elastic_space {3};

int main(int argc, char * argv[]) {

    float u = 1;
    if (argc > 1) u = atof(argv[1]);

    //Task takes T_min, T_max, C, E
    elastic_space.add_task(Task {5,6,0.3,3});
    elastic_space.add_task(Task {12,17,0.7,4});
    elastic_space.add_task(Task {23,36,0.1,1});
    elastic_space.generate();

    std::cout << "\nRegion Space:\n";
    for (const Region & region : elastic_space.regions) {
        std::cout << region.lb << ' ' << region.ub << ' ' << (region.chain - elastic_space.chains.data()) << std::endl;
    }
    std::cout << "\nChains:\n";
    for (const Chain & chain : elastic_space.chains) {
        std::cout << chain.u_min << ' ' << chain.u_max << std::endl;
    }

    for (float u = 0.095; u < 0.15; u+=0.001) {

        Chain * chain1 = elastic_space.assign_periods_slow(u);
        if(!chain1) {
            std::cout << "Utilization " << u << " is too low!" << std::endl;
            continue;
        }
        std::cout << "\n";
        print_info(elastic_space.tasks);

        Chain * chain2 = elastic_space.assign_periods(u);
        if(!chain2) {
            std::cout << "Utilization " << u << " is too low!" << std::endl;
            continue;
        }
        std::cout << "\n";
        print_info(elastic_space.tasks);
        
        std::cout << "\nUtilization " << u << " chains " << elastic_space.chains.size() << ' '
                  << (chain1 - elastic_space.chains.data()) << ' ' << (chain2 - elastic_space.chains.data()) << std::endl;


    }
    return 0;
}