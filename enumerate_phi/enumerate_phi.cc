/*

enumerate_phi.cc

This produces results for
Figure 4 of Section V: "The Ordered Harmonic Elastic Problem."

For numbers of tasks from 2--30 and values k from 1--100,
this computes h_{n,k} the number of possible projected harmonic intervals.

From the paper, we have:

h_{n,k} = sum_{l=1}^k h

*/

#include <iostream>
#include <cmath>
#include <numeric>

constexpr int max_tasks = 30;
constexpr int k = 100;

//We skip task 1
long long counts[max_tasks-1][k];


/*
    recurse_ind(const int prod, const int task, const int n_tasks)

    Recursively project harmonic intervals.

    We have a system of n_tasks
    At task n-1 we have already achieved a product of prod
    In other words,
    a_1 * a_2 * a_3 * ... * a_{n-1} = prod
    (where a_1 = 1)

    We can now project onto a_n and further to the end

*/
void recurse_ind(const int prod, const int n, const int n_tasks) {

    //Product of multipliers exceeds k,
    //Do not keep projecting
    if(prod > k) return;

    for (int a_n = 1; a_n <= k; ++a_n) {
        //prod = a_1 * a_2 * a_3 * ... * a_{n-1}
        const int l = prod * a_n;

        //Increase count for h*_{n,l}
        //(Index at n-2 because we skip task 1)
        if (l <= k) counts[n-2][l-1]++;

        //Project onto the next task
        if (n < n_tasks) recurse_ind(l, n+1, n_tasks);

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

    
    //Recursively compute all values h*_{n,l} for a given n
    //We start by projecting from task 1 to task 2
    //The product prod = a_1 is 1, since a_1 is by definition 1
    recurse_ind(1, 2, max_tasks);    

    //Compute h_{n,k} by summing over h* values
    for (int n_tasks = 2; n_tasks <= max_tasks; ++n_tasks) {
        for (int l = 1; l < k; ++l) {
            counts[n_tasks-2][l] += counts[n_tasks-2][l-1];
        }
    }

    //Print results
    for (int n_tasks = 2; n_tasks <= max_tasks; ++n_tasks) {
        for (int l = 1; l<=k; ++l) {
            std::cout << n_tasks << ' ' << l << ' ' << counts[n_tasks-2][l-1] << std::endl;
        }
    }
}

int main(int argc, char * argv[]) {

    get_harmonic_counts();

}