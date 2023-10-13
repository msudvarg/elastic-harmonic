# elastic-harmonic

task.cc: Defines task related functionality

harmonic.cc: Functionality for the harmonic elastic model

scheduler.cc: A basic wrapper to add tasks and assign utilizations. For now it's just in testing mode (add tasks, assign a utilization, print info). We'll need to think about how to make it into something that's actually usable.

Basic idea of scheduler:

It has a struct elastic_space of type Harmonic_Elastic encapsulating the scheduling state and functionality.

* Initilize elastic_space with the number of tasks.
* Add tasks using elastic_space.add_task(Task {T_min, T_max, C, E});
* After adding tasks, call elastic_space.generate() to do the offline scheduling computation
* Update task periods by calling elastic_space.assign_periods_slow(u), where u is the schedulable utilization bound
* Fast version coming soon!