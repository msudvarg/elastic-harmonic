#include "harmonic.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <array>
#include <list>
#include <algorithm>

//Project a region from interval i-1 to interval i
//Any parts of the region overlapping with the overlap between i-1 and i
//can be carried forward exactly, rather than having the multiples projected up.
Harmonic_Projection find_harmonic(const Tasks & taskset, std::vector<Harmonic_Projection> sources, const int i) {
    

    //End of the chain, no more forward projections
    if (i >= (int)taskset.size()) {
        if (sources.size()) return sources[0];
        else return {{0,0}, {0}};
    }

    const Interval & target = taskset[i].i;

    std::vector<Harmonic_Projection> targets;

    int overlap_index = -1;
    float overlap_min = taskset.back().i.t_max;

    //Loop through sources
    for (size_t i = 0; i < sources.size(); ++i) {
        const Interval & source = sources[i].i;

        //There is overlap
        if (target.t_min < source.t_max) {

            auto multiples = sources[i].multiples;
            multiples.push_back(1);

            //Overlap carried forward exactly rather than being projected up
            targets.push_back({Interval {target.t_min, source.t_max}, multiples});

            //Check if this is the maximum overlap split
            if (source.t_min < overlap_min) {
                overlap_min = source.t_min;
                overlap_index = i;
            }
        }

        //There is no overlap
        else {
            int lb = (int) std::ceil(target.t_min/source.t_max);
            int ub = (int) std::floor(target.t_max / source.t_min);

            //Forward project over multipliers
            for (int a = lb; a <= ub; ++a) {
                Interval overlap;
                auto multiples = sources[i].multiples;
                multiples.push_back(a);

                overlap.t_min = std::max(source.t_min * a, target.t_min);
                overlap.t_max = std::min(source.t_max * a, target.t_max);
                targets.push_back({overlap, multiples});
            }
        }
    }

    //There was overlap, forward project split
    if (overlap_index > -1) {
        Interval overlap_split = {overlap_min, target.t_min};
        int lb = 2;
        int ub = (int) std::floor(target.t_max / overlap_split.t_min);
        for (int a = lb; a <= ub; ++a) {
            Interval overlap;
            auto multiples = sources[i].multiples;
            multiples.push_back(a);

            overlap.t_min = std::max(overlap_split.t_min * a, target.t_min);
            overlap.t_max = std::min(overlap_split.t_max * a, target.t_max);
            targets.push_back({overlap, multiples});
        }

    }

    return find_harmonic(taskset, targets, i+1);
}


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
    if(U > chain.u_max) return chain.O_min;
    if(U < chain.u_min) return std::numeric_limits<float>::max();
    return chain.A * U * U - chain.B * U;
}

//For a given chain and utilization, compute the corresponding loss
float compute_loss_with_bounds(const Chain & chain, const float U) {
    if(U > chain.u_max) return chain.O_min;
    if(U < chain.u_min) return std::numeric_limits<float>::max();
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

/*
void generate_intersections(float u_)

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
*/

bool operator < (const Region & a, const float u) {
    return a.ub < u;
}

Chain * get_min(float lb, float ub, Chain * a, Chain * b) {
    float u = (lb+ub)/2;
    float loss_a = compute_loss_with_bounds(*a, u);
    float loss_b = compute_loss_with_bounds(*b, u);
    if (loss_a < loss_b) return a;
    else return b;
}

void Harmonic_Elastic::generate_intersections() {

    std::list<Region> region_list;

    //Insert first chain as only region
    Chain & chain = chains[0];
    region_list.push_back(Region {chain.u_min, std::numeric_limits<float>::max(), &chain});

    //Insert all other chains
    for (size_t i = 1; i < chains.size(); ++i) {
        Chain & insert = chains[i];

        //Check if lower bound of chain is less than current lowest bound
        std::list<Region>::iterator it = region_list.begin();
        if(insert.u_min < it->chain->u_min) {
            region_list.push_front(Region {insert.u_min, it->chain->u_min, &insert});
        }

        //Intersect with all chains already inserted
        while (it != region_list.end()) {

            //Skip those regions where the upper bound on utilization is less than the new chain's lower bound
            while(it->ub <= insert.u_min) ++it;
            
            Chain & existing = *it->chain;

            //New chain's minimum utilization is greater than lower bound of region,
            //so split region
            if (insert.u_min > it->lb) {
                region_list.insert(it, Region {it->lb, insert.u_min, &existing});
                it->lb = insert.u_min;
            }

            //Pointer to the previous region -- track this so we can merge
            std::list<Region>::iterator merge_into;
            if (it == region_list.begin()) merge_into = region_list.end(); //There is no previous region
            else merge_into = std::prev(it);

            //Find intersections
            Chain * line; Chain * parabola;
            float q=0; //Parabola/parabola intersection
            float l=0; //Line/parabola intersection

            //Which is line and which is parabola in line/parabola intersection
            if (insert.O_min < existing.O_min) {
                line = &existing;
                parabola = &insert;
            }
            else {
                line = &insert;
                parabola = &existing;
            }

            //Calculate parabola/parabola intersection
            if (parabola->A != line->A && parabola->B != line->B) {
                q = (parabola->B - line->B)/(parabola->A - line->A);
                if(q <= it->lb || q >= it->ub || q > parabola->u_max || q > line->u_max) {
                    q = 0;
                }
            }

            //Calculate parabola/line intersection
            {
                float A = parabola->A;
                float B = parabola->B;
                float O = line->O_min;
                l = (B - std::sqrt(B+4*A*O))/(2*A);
                if(l <= it->lb || l >= it->ub || l < line->u_max || l > parabola->u_max) {
                    l = 0;
                }
            }

            //Add regions
            float new_lb;
            if (q > 0 && l > 0) {
                new_lb = std::max(l,q);
                region_list.insert(it, Region {it->lb, std::min(l,q),
                    get_min(it->lb, std::min(l,q), &insert, &existing)} );
                region_list.insert(it, Region {std::min(l,q), new_lb,
                    get_min(std::min(l,q), new_lb, &insert, &existing)} );
            }
            else if (q > 0 || l > 0) {
                new_lb = std::max(l,q);
                region_list.insert(it, Region {it->lb, new_lb,
                    get_min(it->lb, new_lb, &insert, &existing)} );
            }
            else {
                new_lb = it->lb;
            }
            it->lb = new_lb;
            it->chain = get_min(new_lb, it->ub, &insert, &existing);

            //Merge from previous region to end of new set of regions
            if (merge_into == region_list.end()) merge_into = region_list.begin();
            ++merge_into;
            while (merge_into != std::next(it)) {
                std::list<Region>::iterator prev = std::prev(merge_into);
                if (merge_into->chain == prev->chain) {
                    merge_into->lb = prev->lb;
                    region_list.erase(prev);
                }
                ++merge_into;
            }

            ++it;


            /* Old implementation:
            //Check if both chains have equal objectives.
            //If so, replace the old one with the new one if the new one's u_max is larger
            //and the existing chain's u_max is within the region's upper bound.
            //And that's it for these regions
            if (insert.A == existing.A && insert.B == existing.B) {
                if(insert.u_max > existing.u_max && existing.u_max < it->ub) {

                    //Check if new chain's lower bound is greater than region's lower bound.
                    //If so, split region at new chain's lower bound
                    //and insert a new region with the new chain
                    if (insert.u_min > it->lb) {
                        region_list.insert(next, Region {insert.u_min, it->ub, &insert});
                        it->ub = insert.u_min;
                    }
                    //Otherwise, replace the old chain directly
                    else {
                        it->chain = &insert;
                    }
                }

                continue;

            }


            //Check which chain is lower right after U=0
            //It's the one with the larger B value, though if B values are equal,
            //it's the one with the smaller A value.
            Chain * lower; Chain * upper;
            if (insert.B == existing.B) {
                lower = (insert.A < existing.A ? &insert : &existing);
                lower = (insert.A < existing.A ? &existing : &insert);
            }
            else {
                lower = (insert.B > existing.B ? &insert : &existing);
                upper = (insert.B > existing.B ? &existing : &insert);
            }

            //If A1=/=A2 and B1=/=B2, check where the quadratic objectives intersect
            //If they don't intersect, set to infinity
            float quadcept = std::numeric_limits<float>::max();
            if(insert.A != existing.A && insert.B != existing.B) {
                quadcept = (insert.B - existing.B)/(insert.A - existing.A);
            }

            //Check if the lower chain's horizontal (o_max)
            //intercepts the upper chain before the quadratic intersection
            if(lower->u_max < quadcept) {
                //Calculate the intercept
                float intercept = (upper->B - sqrtf(upper->B*upper->B + 4*upper->A*lower->O_min))/(-2*lower->O_min);
                if (intercept < quadcept && intercept < it->ub && intercept > it->lb) {
                    //lower left of intercept, upper right of intercept
                    //done, move on
                }
            }
            else {
                //Check if the quadratic intersection happens in the region
                

                //If so, check if the upper chain's horizontal (o_max)
                //intercepts the lower chain after the intersection, but before the upper bound
            }

            */


        }

    }

    //Copy from list to vector for easy search
    regions.reserve(region_list.size());
    regions.insert(regions.begin(), std::make_move_iterator(std::begin(region_list)), std::make_move_iterator(std::end(region_list)));

}

//Assign periods based on utilization
void Harmonic_Elastic::assign_periods(const Chain & chain, float u_max) {
    if(u_max >= chain.u_max) {
        u_max = chain.u_max;
    }
    tasks[0].t = chain.Y/u_max;
    for (size_t i = 1; i < chain.harmonics.size(); ++i) {
        tasks[i].t = tasks[0].t * chain.harmonics[i].a;
    }
}

Chain * Harmonic_Elastic::assign_periods(float u_max) {

    if (u_max < u_min) return nullptr;

    std::vector<Region>::iterator region = std::lower_bound(regions.begin(), regions.end(), u_max);

    //Shouldn't ever happen, but just in case
    if(region == regions.end()) return nullptr;

    assign_periods(*region->chain, u_max);
    
    return region->chain;


}


Chain * Harmonic_Elastic::assign_periods_slow(float u_max) {

    if(u_max < u_min) return nullptr;

    float min_loss = std::numeric_limits<float>::max();
    int index = -1;

    //Check against all chains
    for (size_t i = 0; i < chains.size(); ++i) {
        const Chain & chain = chains[i];

        //If utilization is less than chain can accomodate, skip        
        if(u_max < chain.u_min) continue;

        float loss = compute_loss_with_bounds(chain, u_max);

        if(loss < min_loss) {
            min_loss = loss;
            index = i;
        }
    }

    //Shouldn't ever happen, but just in case
    if(index < 0) return nullptr;

    assign_periods(chains[index], u_max);
    
    
    return &chains[index];
}


Harmonic_Elastic::Harmonic_Elastic(int n_tasks) {
    tasks.reserve(n_tasks);
}

void Harmonic_Elastic::add_task(Task t) {
    tasks.push_back(t);
}

void Harmonic_Elastic::generate() {
    chains = enumerate_harmonics(tasks, {tasks[0].i, 1}, 1, {});

#ifdef DEBUGINFO
    print_harmonics(chains);
#endif

    backpropagate(chains);
    u_min = compute_chain_properties(tasks, chains);
    generate_intersections();
    
#ifdef DEBUGINFO
    std::cout << '\n';
    print_harmonics(chains);
    std::cout << '\n';
    std::cout << "U_min " <<  u_min << std::endl;
    print_info(chains);
    std::cout << '\n';
    print_info(tasks);
#endif
}