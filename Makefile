all: scheduler.cc harmonic.cc task.cc
	g++ scheduler.cc harmonic.cc task.cc -o harmonic -Wall