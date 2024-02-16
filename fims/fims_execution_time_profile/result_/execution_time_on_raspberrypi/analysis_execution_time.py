import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_name = 'sparse'
file_name2 = 'dense'
path = ['/image_process_cost_user.txt', '/inversion_time_user.txt', '/hk_reading_time_user.txt']
data = []
for p in path:
    data1 = []
    for i in range(1, 11, 1):
        file = 'run_time_' + file_name + '_' + str(i) + p
        # print(file)
        cur_data = np.loadtxt(file)
        # print(f'max index: {cur_data.argmax()}  max value: {cur_data[cur_data.argmax()]}')
        data1 = np.concatenate((data1, cur_data[1:]))
        if p == '/image_process_cost_user.txt':
            file = 'run_time_' + file_name2 + '_' + str(i) + p
            cur_data = np.loadtxt(file)
            # print(f'max index: {cur_data.argmax()}  max value: {cur_data[cur_data.argmax()]}')
            data1 = np.concatenate((data1, cur_data[1:]))
    data.append(data1)


path_sys = ['/image_process_cost_sys.txt', '/inversion_time_sys.txt', '/hk_reading_time_sys.txt']
data2 = []
for p in path_sys:
    data1 = []
    for i in range(1, 11, 1):
        file = 'run_time_' + file_name + '_' + str(i) + p
        # print(file)
        cur_data = np.loadtxt(file)
        # print(f'max index: {cur_data.argmax()}  max value: {cur_data[cur_data.argmax()]}')
        data1 = np.concatenate((data1, cur_data[1:]))
        if p == '/image_process_cost_sys.txt':
            file = 'run_time_' + file_name2 + '_' + str(i) + p
            cur_data = np.loadtxt(file)
            # print(f'max index: {cur_data.argmax()}  max value: {cur_data[cur_data.argmax()]}')
            data1 = np.concatenate((data1, cur_data[1:]))

    data2.append(data1)


for i in range(len(data)):
    data[i] = data[i] + data2[i]

components = ['image_processing', 'data_inversion', 'hk_reading']
for i in range(len(data)):
    average = np.mean(data[i])
    print(f'{components[i]}, Average: {average}')
    maximun = np.max(data[i])
    print(f'{components[i]} Maximun: {maximun}')

# Define a palette
colors = sns.color_palette("deep", 5)

# Use the fivethirtyeight style
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(4.6, 2.5))

for i, d in enumerate(data):
    ax.hist(d, bins=30, density=True, alpha=0.8, edgecolor='black', linewidth=1.5, color=colors[i])
    ax.set_yscale('log')
    ax.set_xlabel('Execution Time (ms)', )
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    fig.savefig(f'{components[i]}_execution_time.png')
    # plt.show() 
    ax.clear()


