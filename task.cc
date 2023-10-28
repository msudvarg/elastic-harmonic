#include "task.h"

#include <iostream>

//Print task system information
void print_info(const Tasks & tasks) {
    
    //Format is:
    //T_i^min, T_i^max, T_i, C_i, E_i, U_i^min, U_i^max, U_i
    for(Task task : tasks) {
        std::cout << task.i.t_min << ' ' << task.i.t_max << ' '
                  << task.t << ' ' << task.c << ' ' << task.e << ' '
                  << task.u_min() << ' ' << task.u_max() << ' ' << task.u()
                  << std::endl;
    }
}