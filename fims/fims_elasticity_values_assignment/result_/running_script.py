import subprocess
import os
import signal
import time
import psutil
import json


# def kill_process_tree(pid, including_parent=True):  
#     parent = psutil.Process(pid)
#     for child in parent.children(recursive=True):
#         child.kill()
#     if including_parent:
#         parent.kill()

# Function to run the C++ program
def run_cpp_programs(command, executable):
    cmd = list(command)
    cmd.append(executable)

    proc1 = subprocess.Popen(cmd)

    # Wait for proc1 to complete
    proc1.communicate()
    return proc1.returncode == 0

# Function to rename the folder and create a new one
def manage_folders(old_folder, new_folder, arg):
    os.rename(old_folder, f"{old_folder}_{arg}")

    # Create a new folder for the next run
    os.makedirs(new_folder, exist_ok=True)


def manage_files(old_file, arg1, arg2):
    name_part, extension_part = os.path.splitext(old_file)
    new_name = f"{name_part}_{arg1}_{arg2}{extension_part}"
    os.rename(old_file, new_name)
    
    # Create a new file as a placeholder (if specific handling is needed)
    # open(new_file, 'a').close()
    



if __name__=="__main__": 

    fims_path = '../build/'
    fims_exe = 'fims'
    result_file = "n_Dp_fixed.txt"

    T_image = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    T_HK = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    T_inversion = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    command = ["sudo"]
    fims_config_file_path = '../configuration/config.json'

    # # get result for expanding image process duretion from 100 to 1000
    # for i in range(len(T_image)): 
    #     with open(fims_config_file_path, 'r') as file:
    #         data = json.load(file)
    #         data['image_processing_duration'] = T_image[i] 
    #         data['HK_reading_duration'] = 500
    #         data['data_inversion_duration'] = 1000

    #     with open(fims_config_file_path, 'w') as file:
    #         json.dump(data, file, indent=4)
        
    #     fims_executable = fims_path + fims_exe

    #     if run_cpp_programs(command, fims_executable):
    #         print(f"Running program with image process duration {T_image[i]}: Success")
    #         manage_files(result_file, 'image', T_image[i])
    #     else:
    #         print(f"Running program with image process duration {T_image[i]}: Fail")
    #     time.sleep(1)  # pause for 1 second between runs


    for i in range(len(T_HK)): 
        with open(fims_config_file_path, 'r') as file:
            data = json.load(file)
            data['HK_reading_duration'] = T_HK[i] 
            data['image_processing_duration'] = 100 
            data['data_inversion_duration'] = 1000

        with open(fims_config_file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        fims_executable = fims_path + fims_exe

        if run_cpp_programs(command, fims_executable):
            print(f"Running program with hk reading duration {T_HK[i]}: Success")
            manage_files(result_file, 'hk', T_HK[i])
        else:
            print(f"Running program with hk reading duration {T_HK[i]}: Fail")
        time.sleep(1)  # pause for 1 second between runs

    
    for i in range(len(T_inversion)): 
        with open(fims_config_file_path, 'r') as file:
            data = json.load(file)
            data['data_inversion_duration'] = T_inversion[i] 
            data['image_processing_duration'] = 100 
            data['HK_reading_duration'] = 500

        with open(fims_config_file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        fims_executable = fims_path + fims_exe

        if run_cpp_programs(command, fims_executable):
            print(f"Running program with hk reading duration {T_inversion[i]}: Success")
            manage_files(result_file, 'inversion', T_inversion[i])
        else:
            print(f"Running program with hk reading duration {T_inversion[i]}: Fail")
        time.sleep(1)  # pause for 1 second between runs