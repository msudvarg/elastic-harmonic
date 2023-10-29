#include "harmonic.h"

#include <iostream>

Harmonic_Elastic elastic_space {3};

int main(int argc, char * argv[]) {

    float u = 1;
    if (argc > 1) u = atof(argv[1]);

    //Task takes T_min, T_max, C, E
    elastic_space.add_task(Task {5, 20, 0.0015, 0.0025682});
    elastic_space.add_task(Task {50, 150, 31.3, 378.19756});
    elastic_space.add_task(Task {50, 1000, 354.55, 765098.615});

    if(!elastic_space.generate()) {
        std::cout << "Could not make harmonic assignment!" << std::endl;
        return 0;
    }

    std::cout << "Minimum utilization accommodated: " << elastic_space.get_u_min() << std::endl;

    Chain * chain1 = elastic_space.assign_periods_slow(u);
    if(!chain1) {
        std::cout << "Utilization " << u << " is too low!" << std::endl;
        return 0;
    }
    print_info(elastic_space.get_tasks());

    return 0;
}