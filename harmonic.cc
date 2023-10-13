#include "harmonic.h"

#include <iostream>
#include <cmath>
#include <limits>

//Get all harmonic chains for a set of intervals
//This is a recursive, depth-first-search projection of harmonic regions
//Takes a set of tasks, the current harmonic being projected,
//an index (which interval are we projecting to), and a harmonic chain
std::vector<Chain> enumerate_harmonics(const Tasks & taskset, const Harmonic harmonic, const int i, Chain chain) {

    //Push back the current harmonic into the chain
    chain.harmonics.push_back(harmonic);

    //No more intervals to forward project
    if (i == (int)taskset.size()) return {chain};

    //Vector of chains
    std::vector<Chain> chains;

    //Get multipliers from current interval to next task's period interval
    Interval l = harmonic.i;
    Interval r = taskset[i].i;
    int lb = (int) std::ceil(r.t_min/l.t_max);
    int ub = (int) std::floor(r.t_max/l.t_min);

    //Iterate over multipliers
    for (int a = lb; a <= ub; ++a) {

        //Get overlapping projection region
        Interval overlap;
        overlap.t_min = std::max(l.t_min * a, r.t_min);
        overlap.t_max = std::min(l.t_max * a, r.t_max);
        Harmonic h {overlap, a};

        //Recursively project harmonic region to end
        auto new_chains = enumerate_harmonics(taskset, h, i+1, chain);
        for (auto new_chain: new_chains) {
            chains.push_back(new_chain);
        }
    }

    return chains;

}


//Forward projection creates overlapping regions with the target interval
//But the projection from the source interval may contain a region that does not overlap the target
//So we have to go backward from the end to trim source intervals so all projections fully overlap the allowed intervals
void backpropagate(std::vector<Chain> & chains) {

    //Loop over all harmonic chains
    for (auto & full_chain : chains) {
        auto & chain = full_chain.harmonics;

        //Walk backward from the end of the chain
        for(int i = chain.size() - 1; i > 0; --i) {

            //Trim the regions
            Interval & l = chain[i-1].i;
            Interval & r = chain[i].i;
            l.t_min = r.t_min / chain[i].a;
            l.t_max = r.t_max / chain[i].a;
        }

        //Also, each projection is defined by its multiplier from source to target,
        //but our algorithm requires the multiplier from base task tau_1 to target
        //so update the multipliers accordingly
        for(size_t i = 1; i < chain.size(); ++i) {
            chain[i].a *= chain[i-1].a;
        }
    }
}

//Print the harmonic projections
void print_harmonics(const std::vector<Chain> & harmonics) {
    
    for(auto chain : harmonics) {
        for (Harmonic h : chain.harmonics) {
            std::cout << h.a << ' ' << h.i.t_min << ' ' << h.i.t_max << " | ";
        }
        std::cout << std::endl;
    }
}

//Print the utilization limit and objective information about a corresponding projection
void print_info(const std::vector<Chain> & harmonics) {
    

    for(auto chain : harmonics) {
        std::cout << chain.u_min << ' ' << chain.u_max << ' '
                  << chain.O_min << ' ' << chain.O_max << ' '
                  << chain.A << ' ' << chain.B << std::endl;
    }
}

//For a given chain and utilization, compute the corresponding loss
float compute_loss(const Chain & chain, const float U) {
    return chain.A * U * U - chain.B * U;
}

//A slower way to compute the loss, used to verify chain properties
float compute_loss(const Tasks & taskset, const Chain & chain, const float U) {
    float T = 0;
    for (size_t i = 0; i < taskset.size(); ++i) {
        T += taskset[i].c/(float)chain.harmonics[i].a;
    }
    T /= U;

    float O = 0;
    for (size_t i = 0; i < taskset.size(); ++i) {
        float o = taskset[i].u_max() - taskset[i].c / (chain.harmonics[i].a * T);
        O += o * o / taskset[i].e;
    }

    return O;
}

//Compute the constant values used to quickly compute loss for a given chain
float compute_chain_properties(const Tasks & taskset, std::vector<Chain> & chains) {

    float u_min_min = std::numeric_limits<float>::max();

    for (Chain & chain : chains) {

        float u_min = 0;
        float u_max = 0;

        float Y = 0;
        float A = 0;
        float B = 0;

        for (size_t i = 0; i < chain.harmonics.size(); ++i) {
            u_min += taskset[i].c/chain.harmonics[i].i.t_max;
            u_max += taskset[i].c/chain.harmonics[i].i.t_min;
            
            float y = taskset[i].c/(float)chain.harmonics[i].a;
            Y += y;
            float u_i_max = taskset[i].u_max();
            A += y * y / taskset[i].e;
            B += 2 * y * u_i_max / taskset[i].e;
        }

        float X = 1/Y;
        A = A * X * X;
        B = B * X;

        chain.A = A;
        chain.B = B;
        chain.Y = Y;
        chain.u_min = u_min;
        chain.u_max = u_max;


        //Assumes the minimum objective corresponds with maximum utilization
        //TODO: May not be the case if c depends on harmonic range
        chain.O_max = compute_loss(chain, u_min);
        chain.O_min = compute_loss(chain, u_max);

        u_min_min = std::min(u_min_min, u_min);

    }

    return u_min_min;
}

void generate_intersections(float u_max, const std::vector<Chain> & chains) {
    std::vector<Region> regions;
    const Chain & chain = chains[0];
    regions.push_back({0, chain.u_min, true, chain.O_max, 0});
    regions.push_back({chain.u_min, chain.u_max, false, chain.A, chain.B});
    regions.push_back({chain.u_max, u_max, true, chain.O_min, 0});

    // for(int i = 1; i < chains.size(); ++i) {
    //     const Chain & chain = chains[i];
        
    // }
}

void intersections(struct Region, Chain & chain) {

}

bool Harmonic_Elastic::assign_periods_slow(float u_max) {

    if(u_max < u_min) return false;

    float min_loss = std::numeric_limits<float>::max();
    int index = -1;

    //Check against all chains
    for (size_t i = 0; i < chains.size(); ++i) {
        const Chain & chain = chains[i];

        //If utilization is less than chain can accomodate, skip        
        if(u_max < chain.u_max) continue;

        //Otherwise, compute loss
        float loss;

        //If utilization bound is greater than necessary for chain, use its minimum loss
        if(u_max >= chain.u_max) loss = chain.O_min;

        //Otherwise, compute
        else loss = compute_loss(chain, u_max);

        if(loss < min_loss) {
            min_loss = loss;
            index = i;
        }
    }

    //Shouldn't ever happen, but just in case
    if(index < 0) return false;

    //Assign periods based on utilization
    const Chain & chain = chains[index];
    if(u_max >= chain.u_max) {
        u_max = chain.u_max;
    }
    tasks[0].t = chain.Y/u_max;
    for (size_t i = 1; i < chain.harmonics.size(); ++i) {
        tasks[i].t = tasks[0].t * chain.harmonics[i].a;
    }
    
    return true;
}


Harmonic_Elastic::Harmonic_Elastic(int n_tasks) {
    tasks.reserve(n_tasks);
}

void Harmonic_Elastic::add_task(Task t) {
    tasks.push_back(t);
}

void Harmonic_Elastic::generate() {
    chains = enumerate_harmonics(tasks, {tasks[0].i, 1}, 1, {});
    print_harmonics(chains);
    backpropagate(chains);
    std::cout << '\n';
    print_harmonics(chains);
    std::cout << '\n';
    u_min = compute_chain_properties(tasks, chains);
    print_info(chains);
    std::cout << '\n';
    print_info(tasks);
}