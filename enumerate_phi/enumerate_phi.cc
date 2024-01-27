/*

enumerate_phi.cc

This produces results for
Figure 4 of Section V: "The Ordered Harmonic Elastic Problem."

For numbers of tasks from 1--30 and values k from 1--100,
this computes h_{n,k} the number of possible projected harmonic intervals.

From the paper, we have:

h_{n,k} = sum_{l=1}^k h

*/

#include <iostream>
#include <cmath>
#include <numeric>
#include <omp.h>

constexpr int max_tasks = 30;
constexpr int k = 100;

long long counts[max_tasks][k];


/*
    recurse_ind(const int prod, const int task, const int n_tasks)

    Recursively project harmonic intervals.

    We have a system of `n_tasks`
    At task number `task`-1 we have already achieved a product of `prod`
    In other words,
    a_2 * a_3 * ... * a_{task-1} = prod

    We can now project onto a_task and further to the end

*/
void recurse_ind(const int prod, const int task, const int n_tasks) {

    //End condition: product of multipliers exceeds k
    if(prod > k) return;

    for (int i = 1; i <= k; ++i) {
        //prod = a_2 * a_3 * ... * a_{task-1}
        //x = prod * i, where i represents a_task
        const int x = i * prod;

        //We are at the last task
        if (task == n_tasks) {

            //As long as the total product x does not exceed k,
            //Increase the count of ways to achieve a multiple equal to x
            if (x <= k) counts[task-1][x-1]++;
            else return;
        }

        //We are not at the last task
        else {
            
            //Continue to the end
            recurse_ind(x, task+1, n_tasks);
        }
    }
}

/*
    get_harmonic_counts()

    Computes the number of projected harmonic intervals for values of n and k.
    Assumes all tasks have a minimum period of 1 to maximize the number of intervals.

    We call h*_{n,l} the number of ways to generate a sequence of positive integer multiples:
    a_2 * a_3 * ... * a_n = l

    Then the number of projected harmonic intervals h_{n,k} is
    sum_{l=1}^k h*_{n,l}

    For all values of n and l from 1--k, we recursively compute h*_{n,l}

    Then we compute h_{n,k} by summing over the h* values.

*/
void get_harmonic_counts() {

    //Run in parallel for better performance
    #pragma omp parallel
    {
        int begin = omp_get_thread_num();
        int step = omp_get_num_threads();
        for (int n_tasks = begin; n_tasks < max_tasks; n_tasks += step) {

            //Recursively compute all values h*_{n,l} for a given n
            //(We use n_tasks+1 due to 0-indexing)
            recurse_ind(1, 1, n_tasks+1);

            //Compute h_{n,k} by summing over h* values
            for (int i = 1; i < k; ++i) {
                counts[n_tasks][i] += counts[n_tasks][i-1];
            }
        }
    }

    //Print results
    for (int n_tasks = 0; n_tasks < max_tasks; ++n_tasks) {
        for (int i = 0; i<k; ++i) {
            std::cout << n_tasks+1 << ' ' << i+1 << ' ' << counts[n_tasks][i] << std::endl;
        }
    }
}

int main(int argc, char * argv[]) {

    get_harmonic_counts();

}