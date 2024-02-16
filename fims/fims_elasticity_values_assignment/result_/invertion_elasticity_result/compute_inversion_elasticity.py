import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity



def read_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = list(map(float, line.strip().split(',')))
            data.append(row)
    return data


def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = []
    row = []
    for line in lines:
        value = line.strip()
        if value == "NaN":
            matrix.append(row)
            row = []
        else:
            row.append(float(value))

    if row:
        matrix.append(row)

    matrix = np.array(matrix)
    return matrix


def sum_every_n_rows(arr, n):
    num_rows_to_keep = (len(arr) // n) * n
    trimmed_arr = arr[:num_rows_to_keep]   
    reshaped = trimmed_arr.reshape(-1, n, arr.shape[1])
    return reshaped.sum(axis=1)


n_Dp_fixed_matlab = read_csv('n_Dp_fixed_mat.csv')
n_Dp_fixed_matlab = np.array(n_Dp_fixed_matlab, dtype=float)
n_Dp_fixed_matlab = np.nan_to_num(n_Dp_fixed_matlab)

bin_stacks = ['1sbin', '2sbin', '3sbin', '4sbin', '5sbin', '6sbin', '7sbin', '8sbin', '9sbin', '10sbin']

n_fixed_array_matlab = []
time = []
for i in range(len(bin_stacks)):
    n_fixed_array_matlab_ = sum_every_n_rows(n_Dp_fixed_matlab, i+1)
    n_fixed_array_matlab.append(n_fixed_array_matlab_)
    

n_fixed_array_cpp = []
for i in range(len(bin_stacks)):
    n_Dp_fixed_cpp_path = 'n_Dp_fixed_cpp_' + bin_stacks[i] + '.txt'
    n_Dp_fixed_cpp = read_txt(n_Dp_fixed_cpp_path)
    n_Dp_fixed_cpp = np.array(n_Dp_fixed_cpp, dtype=float)
    n_Dp_fixed_cpp = np.nan_to_num(n_Dp_fixed_cpp)
    n_fixed_array_cpp.append(n_Dp_fixed_cpp)

n_fixed_array_cpp_norm = []
n_fixed_array_matlab_norm = []
cos_similarities = []
mean_cos_similarities = []

for i in range(len(bin_stacks)):
    n_fixed_array_cpp_sum = np.sum(n_fixed_array_cpp[i], axis=1)[:, np.newaxis]
    n_fixed_array_cpp_norm_ = np.where(n_fixed_array_cpp_sum!=0, n_fixed_array_cpp[i] / n_fixed_array_cpp_sum, np.nan)
    n_fixed_array_cpp_norm_ = np.nan_to_num(n_fixed_array_cpp_norm_) + 1e-15
    n_fixed_array_cpp_norm.append(n_fixed_array_cpp_norm_)

    n_fixed_array_matlab_sum = np.sum(n_fixed_array_matlab[i], axis=1)[:, np.newaxis]
    n_fixed_array_matlab_norm_ = np.where(n_fixed_array_matlab_sum!=0, n_fixed_array_matlab[i] / n_fixed_array_matlab_sum, np.nan)
    n_fixed_array_matlab_norm_ = np.nan_to_num(n_fixed_array_matlab_norm_) + 1e-15
    n_fixed_array_matlab_norm.append(n_fixed_array_matlab_norm_)

    similarities = [cosine_similarity(n_fixed_array_matlab_norm[i][j].reshape(1, -1), n_fixed_array_cpp_norm[i][j].reshape(1, -1))[0][0] for j in range(n_fixed_array_cpp_norm[i].shape[0])]
    cos_similarities.append(similarities)
    mean_similarities = np.mean(similarities)
    mean_cos_similarities.append(mean_similarities)


data_inversion_duration = ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000']

plt.rcParams["font.size"] = 23
plt.figure(figsize=(10, 5))
plt.plot(data_inversion_duration, mean_cos_similarities, marker='o', linestyle='-', color='b')
xticks_locs = plt.xticks()[0][::2] 
plt.xticks(xticks_locs)
plt.xlabel('Data Inversion Duration(ms)')
plt.ylabel('Mean Cosine Similarity')
# plt.title('Cosine Similarity over Data Inversion Duration')
plt.grid(True)
plt.tight_layout()
plt.show() 

# plt.figure(figsize=(10, 5))
# plt.plot(data_inversion_duration, RMSE, marker='o', linestyle='-', color='b')
# plt.xlabel('Data Inversion Duration')
# plt.ylabel('RMSE')
# plt.title('RMSE over Data Inversion Duration')
# plt.grid(True)
# plt.tight_layout()
# plt.show() 

# plt.figure(figsize=(10, 5))
# plt.plot(data_inversion_duration, nRMSE, marker='o', linestyle='-', color='b')
# plt.xlabel('Data Inversion Duration')
# plt.ylabel('NRMSE')
# plt.title('NRMSE over Data Inversion Duration')
# plt.grid(True)
# plt.tight_layout()
# plt.show() 


Duration = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
mean_cos_similarities = np.array(mean_cos_similarities)

R = 1 / Duration * 1000
R_max = R[0]
X = (R_max - R) ** 2
Y = (mean_cos_similarities[0] - mean_cos_similarities) * 1000

X = np.array(X)
Y = np.array(Y)

slope, intercept = np.polyfit(X, Y, 1)
print(f'y = {slope}x + {intercept}')


# To visualize
# plt.scatter(X, Y, color='blue', label='Data points')
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue')
plt.plot(X, slope * X + intercept, color='red', label='Best Fit')
plt.xlabel(r'$(R^{max} - R)^2$') 
plt.ylabel('Error')
plt.tight_layout()
plt.legend()
plt.show()

# worst_case = 49.0678
worst_case = 55.272
e = worst_case ** 2 / slope / 1000.0
print(f'const: {e}')

# 0.972