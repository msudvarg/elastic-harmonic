#include "harmonic.h"

#include <iostream>

Harmonic_Elastic elastic_space {3};

int main(int argc, char * argv[]) {

    float u = 1;
    if (argc > 1) u = atof(argv[1]);

    //Task takes T_min, T_max, C, E
    elastic_space.add_task(Task {6,6,0.3,3});
    elastic_space.add_task(Task {11,18,0.3,3});
    elastic_space.add_task(Task {12,17,0.7,4});
    elastic_space.add_task(Task {23,36,0.1,1});

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