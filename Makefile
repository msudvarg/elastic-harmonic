all: scheduler.cc harmonic.cc task.cc
	g++ scheduler.cc harmonic.cc task.cc -o harmonic -Wall

debug:  scheduler.cc harmonic.cc task.cc
	g++ scheduler.cc harmonic.cc task.cc -o harmonic -Wall -O0 -ggdb

test:  scheduler_test.cc harmonic.cc task.cc
	g++ scheduler.cc harmonic.cc task.cc -o harmonic -Wall -O0 -ggdb

synthetic: synthetic_tasksets.cc harmonic.cc task.cc
	g++ synthetic_tasksets.cc harmonic.cc task.cc -o synthetic -Wall -O3

fims_periods: fims_get_periods.cc harmonic.cc task.cc
	g++ fims_get_periods.cc harmonic.cc task.cc -o fims_periods -Wno-narrowing -O3
