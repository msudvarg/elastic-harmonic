/*

synthetic_tasksets.cc

This produces results for
Figure 8 of Section VI.D: "Evaluation with Larger Synthetic Task Sets"

Run with ./synthetic_tasksets n_per
Generates n_per task sets for each number of tasks from 5--50

*/

#include "harmonic.h"
#include <random>
#include <cstdlib>
#include <set>
#include <algorithm>
#include <iostream>
#include <chrono>

//Range of task set size to generate
constexpr int n_tasks_min = 5;
constexpr int n_tasks_max = 50;
constexpr int n_tasks_step = 1;

//Range of minimum periods
constexpr int t_min = 1;
constexpr int t_max = 100;
//Maximum to scale maximum period from minimum period
constexpr int t_max_scale_limit = 10;

//Random distributions
std::mt19937 prng;
std::uniform_real_distribution<float> d_t(std::log((float)t_min), std::log((float)t_max));
std::uniform_real_distribution<float> d_t_scale(1, t_max_scale_limit);
std::uniform_real_distribution<float> d_w(0,1);

//Generate periods from log-uniform distribution
int gen_period_exponential(int min, int max) {
    return (int)expf(d_t(prng));
}

//Assign utilizations according to UUnisort
std::vector<float> uunisort(int n, float u) {
    std::uniform_real_distribution<double> d(0.0,u);
    std::set<double> boundaries;
    std::vector<float> utilizations;

    for (int i = 0; i < n-1; ++i) {
        boundaries.insert(d(prng));
    }

    float low = 0;
    for (double b : boundaries) {
        utilizations.push_back(b-low);
        low = b;
    }

    utilizations.push_back(u-low);

    return utilizations;

}

//Generate a set of n tasks with total utilization u
std::vector<Task> generate_taskset(int n, float u) {

    //Periods
    std::vector<int> periods;
    for (int i = 0; i < n; ++i) {
        periods.push_back(gen_period_exponential(t_min,t_max));
    }

    //Utilizations
    std::vector<float> utilizations = uunisort(n,u);

    //Assign task parameters
    std::vector<Task> tasks;
    for (int i = 0; i < n; ++i) {

        //Max period
        float t_min = periods[i];
        float t_max = (int)(d_t_scale(prng) * t_min);

        //Execution time
        float c = utilizations[i] * t_min;

        //Elasticity
        float w = d_w(prng);
        float e = c*c/w;

        //Create task
        Task tau {t_min, t_max, c, e};
        tasks.push_back(tau);
    }

    //Sort by minimum period
    std::sort(tasks.begin(), tasks.end());

    return tasks;
}

int main(int argc, char * argv[]) {

    //Number of task sets for each size
    int n_per = atoi(argv[1]);   
    
    std::chrono::high_resolution_clock::time_point t0, t1; 

    //Loop over experimental parameters
    for (int n_tasks = n_tasks_min; n_tasks <= n_tasks_max; n_tasks += n_tasks_step) {
        for (int i = 0; i < n_per; ++i) {

            std::cout << n_tasks << ' ' << i << ' ';

            //Generate task set
            Tasks tasks = generate_taskset(n_tasks, 1);

            //Data structure for elastic scheduling lookup table
            Harmonic_Elastic elastic_space {tasks};

            //Count the number of projected harmonic zones on last task's period interval
            int n_harmonics = 0;
            t0 = std::chrono::high_resolution_clock::now();
            bool found = find_harmonic(tasks, &n_harmonics);
            t1 = std::chrono::high_resolution_clock::now();
            if(found && !verify_harmonic(tasks)) {
                std::cout << "Could not verify harmonics 1!" << std::endl;
                return -1;
            }
            std::cout <<  found << ' ' << n_harmonics << ' '
                      << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << ' ';

            //Only collect additional data if harmonic periods were found
            if(found) {

                //Generate lookup table and count regions and
                //total projected harmonic intervals
                t0 = std::chrono::high_resolution_clock::now();
                bool generated = elastic_space.generate();
                t1 = std::chrono::high_resolution_clock::now();

                if(!generated) {
                    std::cout << "Found harmonics but could not enumerate them!" << std::endl;
                    return -1;
                }

                std::cout << elastic_space.get_num_chains() << ' ' << elastic_space.get_num_regions() << ' '
                          << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << ' ';

                //Use lookup table to adjust utilization
                float u = (1+elastic_space.get_u_min())/2;
                
                t0 = std::chrono::high_resolution_clock::now();
                elastic_space.assign_periods(u);
                t1 = std::chrono::high_resolution_clock::now();
                std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() << ' ';
                if(!verify_harmonic(elastic_space.get_tasks())) {
                    std::cout << "Could not verify harmonics 2!" << std::endl;
                    return -1;
                }
                
                //Use naive enumeration over projected harmonic intervals to adjust utilization
                t0 = std::chrono::high_resolution_clock::now();
                elastic_space.assign_periods_slow(u);
                t1 = std::chrono::high_resolution_clock::now();
                std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << ' ';
                if(!verify_harmonic(elastic_space.get_tasks())) {
                    std::cout << "Could not verify harmonics 3!" << std::endl;
                    return -1;
                }

            }

            std::cout << std::endl;

            


        }
    }
}
