#include "harmonic.h"

#include <iostream>

Harmonic_Elastic elastic_space {3};

int main(int argc, char * argv[]) {

    float u = 1;
    if (argc > 1) u = atof(argv[1]);

    //Task takes T_min, T_max, C, E
    elastic_space.add_task(Task {5,20,0.015,0.263});
    elastic_space.add_task(Task {50,200,31.3,413});
    elastic_space.add_task(Task {50,1200,270,4.94e5});
    elastic_space.generate();

    Tasks tasks = elastic_space.get_tasks();
    if(find_harmonic(tasks)) {
        print_info(tasks);
    }
    else {
        std::cout << "Could not make harmonic assignment!" << std::endl;
    }
    if(!verify_harmonic(tasks)) {
        std::cout << "Harmonic assignment invalid!" << std::endl;
    }

#ifdef DEBUGINFO
    std::cout << "\nRegion Space:\n";
    for (const Region & region : elastic_space.regions) {
        std::cout << region.lb << ' ' << region.ub << ' ' << (region.chain - elastic_space.chains.data()) << std::endl;
    }
    std::cout << "\nChains:\n";
    for (const Chain & chain : elastic_space.chains) {
        std::cout << chain.u_min << ' ' << chain.u_max << std::endl;
    }
#endif

    std::cout << "Minimum utilization accommodated: " << elastic_space.get_u_min() << std::endl;

    for (float u = 0.1; u < 1.0; u+=0.05) {

        Chain * chain1 = elastic_space.assign_periods_slow(u);
        if(!chain1) {
            std::cout << "Utilization " << u << " is too low!" << std::endl;
            continue;
        }
        std::cout << "\n";
        print_info(elastic_space.get_tasks());

#ifdef DEBUGINFO
        Chain * chain2 = elastic_space.assign_periods(u);
        if(!chain2) {
            std::cout << "Utilization " << u << " is too low!" << std::endl;
            continue;
        }
        std::cout << "\n";
        print_info(elastic_space.get_tasks());
        
        std::cout << "\nUtilization " << u << " chains " << elastic_space.chains.size() << ' '
                  << (chain1 - elastic_space.chains.data()) << ' ' << (chain2 - elastic_space.chains.data()) << std::endl;

        if (chain1 != chain2) {
            std::cout << "Problem! Chains selected by slow and fast methods to dnot match!" << std::endl;
            return -1;
        }
#endif


    }
    return 0;
}