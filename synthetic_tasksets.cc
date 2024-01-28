
#include "harmonic.h"
#include <random>
#include <cstdlib>
#include <set>
#include <algorithm>
#include <iostream>
#include <chrono>

constexpr int n_tasks_min = 5;
constexpr int n_tasks_max = 50;
constexpr int n_tasks_step = 1;

constexpr int t_min = 1;
constexpr int t_max = 100;
constexpr int t_max_scale_limit = 10;

//t_min log uniform
//t_max uniform 1-10 * t_min
//Utilization (at t_min) according to uunisort (just fix utilization 1)
//Two experiments: either E uniformly in [0,1] or w uniformly in [0,1]
//How long does it take to find harmonic periods
//How many harmonic projections at end
//How long does it take to generate regions
//How many harmonic chains
//How many regions
//Time for binary search versus enumeration

std::mt19937 prng;
std::uniform_real_distribution<float> d_t(std::log((float)t_min), std::log((float)t_max));
std::uniform_real_distribution<float> d_t_scale(1, t_max_scale_limit);
std::uniform_real_distribution<float> d_w(0,1);

int gen_period_exponential(int min, int max) {
    return (int)expf(d_t(prng));
}

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

    int n_per = atoi(argv[1]);   
    
    std::chrono::high_resolution_clock::time_point t0, t1; 

    for (int n_tasks = n_tasks_min; n_tasks <= n_tasks_max; n_tasks += n_tasks_step) {
        for (int i = 0; i < n_per; ++i) {

            std::cout << n_tasks << ' ' << i << ' ';
            Tasks tasks = generate_taskset(n_tasks, 1);
            Harmonic_Elastic elastic_space {tasks};

            //Find harmonic periods
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

            if(found) {
                t0 = std::chrono::high_resolution_clock::now();
                bool generated = elastic_space.generate();
                t1 = std::chrono::high_resolution_clock::now();

                if(!generated) {
                    std::cout << "Found harmonics but could not enumerate them!" << std::endl;
                    return -1;
                }

                std::cout << elastic_space.get_num_chains() << ' ' << elastic_space.get_num_regions() << ' '
                          << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << ' ';

                float u = (1+elastic_space.get_u_min())/2;
                
                t0 = std::chrono::high_resolution_clock::now();
                elastic_space.assign_periods(u);
                t1 = std::chrono::high_resolution_clock::now();
                std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() << ' ';
                if(!verify_harmonic(elastic_space.get_tasks())) {
                    std::cout << "Could not verify harmonics 2!" << std::endl;
                    return -1;
                }
                
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
