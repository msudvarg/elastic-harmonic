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


n_Dp_fixed_matlab = read_csv('n_Dp_fixed_mat.csv')
n_fixed_array_matlab = np.array(n_Dp_fixed_matlab, dtype=float)
n_fixed_array_matlab = np.nan_to_num(n_fixed_array_matlab)[:1195]

image_stacks = ['hk_1', 'hk_2', 'hk_3', 'hk_4', 'hk_5', 'hk_6', 'hk_7', 'hk_8', 'hk_9', 'hk_10']
n_fixed_array_cpp = []

for i in range(len(image_stacks)):
    n_Dp_fixed_cpp_path = 'n_Dp_fixed_cpp_' + image_stacks[i] + '.txt'
    n_Dp_fixed_cpp = read_txt(n_Dp_fixed_cpp_path)
    n_Dp_fixed_cpp = np.array(n_Dp_fixed_cpp, dtype=float)
    n_Dp_fixed_cpp = np.nan_to_num(n_Dp_fixed_cpp)[:1195]
    n_fixed_array_cpp.append(n_Dp_fixed_cpp)

n_fixed_array_matlab_sum = np.sum(n_fixed_array_matlab, axis=1)[:, np.newaxis]
n_fixed_array_matlab_norm = np.where(n_fixed_array_matlab_sum!=0, n_fixed_array_matlab / n_fixed_array_matlab_sum, np.nan)

n_fixed_array_cpp_norm = []
cos_similarities = []
mean_cos_similarities = []

for i in range(len(image_stacks)):
    n_fixed_array_cpp_sum = np.sum(n_fixed_array_cpp[i], axis=1)[:, np.newaxis]
    n_fixed_array_cpp_norm_ = np.where(n_fixed_array_cpp_sum!=0, n_fixed_array_cpp[i] / n_fixed_array_cpp_sum, np.nan)
    n_fixed_array_cpp_norm_ = np.nan_to_num(n_fixed_array_cpp_norm_) + 1e-15
    n_fixed_array_cpp_norm.append(n_fixed_array_cpp_norm_)
    similarities = [cosine_similarity(n_fixed_array_matlab_norm[j].reshape(1, -1), n_fixed_array_cpp_norm[i][j].reshape(1, -1))[0][0] for j in range(n_fixed_array_cpp_norm[i].shape[0])]
    cos_similarities.append(similarities)
    mean_similarities = np.mean(similarities)
    mean_cos_similarities.append(mean_similarities)


image_process_duration = ['500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000']

plt.rcParams["font.size"] = 23
plt.figure(figsize=(10, 5))
plt.plot(image_process_duration, mean_cos_similarities, marker='o', linestyle='-', color='b')
xticks_locs = plt.xticks()[0][::2] 
plt.xticks(xticks_locs)

plt.xlabel('HK Reading Duration(ms)')
plt.ylabel('Mean Cosine Similarity')
# plt.title('Cosine Similarity over HK Reading Interval')
plt.grid(True)
plt.tight_layout()
plt.show() 

Duration = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
mean_cos_similarities = np.array(mean_cos_similarities)
print(mean_cos_similarities)

R = 1 / Duration * 1000
R_max = R[0]
X = (R_max - R) ** 2
Y = (mean_cos_similarities[0] - mean_cos_similarities) * 1000

X = np.array(X)
Y = np.array(Y)

slope, intercept = np.polyfit(X, Y, 1)
print(f'y = {slope}x + {intercept}')

plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue')
plt.plot(X, slope * X + intercept, color='red', label='Best Fit')
plt.xlabel(r'$(R^{max} - R)^2$') 
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('output_filename.png', format='png', dpi=300, bbox_inches='tight')
# plt.savefig('hk_reading_errorfit.png')

worst_case = 0.747
e = worst_case ** 2 / slope / 1000.0
print(f'const: {e}')





