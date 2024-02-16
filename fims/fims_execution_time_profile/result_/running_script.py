import subprocess
import os
import time
import sys

# Function to run the C++ program
def run_cpp_program(command, executable):
    command.append(executable)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0  

# Function to rename the folder and create a new one
def manage_folders(old_folder, new_folder, sh, i):
    os.rename(old_folder, f"{old_folder}_{sh}_{i}")
    os.makedirs(new_folder, exist_ok=True)


run_times = 1
if len(sys.argv) > 1:
    try:
        run_times = int(sys.argv[1]) 
    except ValueError:
        print("Please provide a valid integer as the argument.")
        sys.exit(1) 

os.system("echo -1 > /proc/sys/kernel/sched_rt_runtime_us")
os.system("cpufreq-set -r -g performance")


executable = "../build/fims"  
result_folder = "run_time"
command = ["sudo"]

for i in range(0, run_times, 1):  # define how many time to run the code
    print(f'start running {i} ...')
    if run_cpp_program(command, executable):
        print(f"Run {i+1}: Success")
        # Rename the old folder and create a new one for the next run
        manage_folders(result_folder, result_folder, 'sparse', i)
    else:
        print(f"Run {i+1}: Failure")

    time.sleep(1)  # Optional: pause for 1 second between runs



