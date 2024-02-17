import subprocess
import os
import signal
import time
import psutil
import json


def kill_process_tree(pid, including_parent=True):  
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    if including_parent:
        parent.kill()

# Function to run the C++ program
def run_cpp_programs(command, executable1, executable2, *args2):
    cmd1 = list(command)
    cmd1.append(executable1)

    cmd2 = list(command)
    cmd2.append(executable2)
    cmd2.extend(args2)  # Add arguments for executable2

    proc1 = subprocess.Popen(cmd1)
    proc2 = subprocess.Popen(cmd2)

    # Wait for proc1 to complete
    proc1.communicate()

    # Kill proc2 when proc1 completes
    kill_process_tree(proc2.pid)

    return proc1.returncode == 0

# Function to rename the folder and create a new one
def manage_folders(old_folder, new_folder, arg):
    # Rename the folder
    os.rename(old_folder, f"{old_folder}_{arg}")

    # Create a new folder for the next run
    os.makedirs(new_folder, exist_ok=True)



if __name__=="__main__": 

    os.system("echo -1 > /proc/sys/kernel/sched_rt_runtime_us")
    os.system("cpufreq-set -r -g performance")

    fims_path = '../build/'
    fims_exe = 'fims'
    overhead_path = '../../over_head/'
    overhead_exe = 'overhead'
    result_folder = 'run_time'

    utilization_D = [0.5, 0.4, 0.3, 0.2, 0.1]
    T_image = [100, 115, 147, 222, 458]
    T_HK = [500, 575, 881, 3325, 3205]
    T_inv = [1000, 2298, 9682, 9973, 9615]

    command = ["sudo"]
    fims_config_file_path = '../configuration/config.json'


    #choose the dataset you want to run by changing the range
    for i in range(len(utilization_D)): 
        with open(fims_config_file_path, 'r') as file:
            data = json.load(file)

        data['HK_reading_duration'] = T_HK[i]  
        data['image_processing_duration'] = T_image[i] 
        data['data_inversion_duration'] = T_inv[i] 

        with open(fims_config_file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        fims_executable = fims_path + fims_exe
        overhead_executable = overhead_path + overhead_exe
        overhead_utilization = 1 - utilization_D[i]
        if run_cpp_programs(command, fims_executable, overhead_executable, str(overhead_utilization)):
            print(f"Running program with available utilization {utilization_D[i]}: Success")
            manage_folders(result_folder, result_folder, utilization_D[i])
        else:
            print(f"Running program with available utilization {utilization_D[i]}: Failure")


        time.sleep(1)  # pause for 1 second between runs

