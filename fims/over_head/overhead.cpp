#include <iostream>
#include <signal.h>
#include <sys/time.h>
#include <chrono>
#include <sched.h> 
#include <unistd.h>
#include <ctime>
#include <cstdlib>

constexpr std::chrono::milliseconds PERIODIC_INTERVAL(50);
std::chrono::milliseconds overhead;


void handler(int signum) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // auto end_time = start_time + (execution_time - overhead);
    auto end_time = start_time + overhead;

    while (std::chrono::high_resolution_clock::now() < end_time) {
        // Busy-wait
    }
}

void setup_itimer() {
    struct itimerval timer;
    timer.it_value.tv_sec = 0;
    timer.it_value.tv_usec = PERIODIC_INTERVAL.count() * 1000; // microseconds
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = PERIODIC_INTERVAL.count() * 1000; // microseconds

    setitimer(ITIMER_REAL, &timer, NULL);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <overhead>\n";
        return 1;
    }

    float overhead_ = std::atof(argv[1]);
    overhead = std::chrono::milliseconds(static_cast<int>(overhead_ * std::chrono::duration_cast<std::chrono::milliseconds>(PERIODIC_INTERVAL).count()));
    std::cout << "overhead: " << overhead.count() << " milliseconds out of 50 milliseconds" << std::endl;


    cpu_set_t mask;
    CPU_ZERO(&mask);     
    CPU_SET(2, &mask);        // Set CPU2
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        std::cerr << "Failed to set CPU affinity" << std::endl;
        return 1;
    }

    struct sched_param param;
    param.sched_priority = 99; // Set priority to 99
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        std::cerr << "Failed to set real-time scheduler" << std::endl;
        return 1;
    }
    
    signal(SIGALRM, handler);
    setup_itimer();

    while (true) {
        pause();
    }

    return 0;
}

