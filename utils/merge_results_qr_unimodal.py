import os
from glob import glob
import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

experiment = 'qr'
data = ['Pancreas']
models_list = ['SimCLR', 'MoCo', 'SimSiam', 'NNCLR', 'BYOL', 'VICReg', 'BarlowTwins']


def compare_model_architectures(data):
    metrics = ['Macro-F1', 'Accuracy']

    cols = []
    for d in data:
        for metric in metrics:
            cols.append(f"{d}_{metric}")
    result = pd.DataFrame(index=models_list, columns=cols)
    std = pd.DataFrame(index=models_list, columns=cols)

    for model in models_list:
        for d in data:
            PATH = experiment + '/' + d + '/' + model
            result_paths = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], f'qr-results.csv'))]
            
            tmp_df = pd.DataFrame(columns=metrics)
            for i, seed_path in enumerate(result_paths):
                with open(seed_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    vals = []
                    for row in reader:
                        vals.append(float(row['0']))
                    tmp_df.loc[i] = [vals[0], vals[1]]
                          
            for k, v in tmp_df.mean(axis=0).to_dict().items():
                result.loc[model, f'{d}_{k}'] = v
            
            for k, v in tmp_df.std(axis=0).to_dict().items():
                std.loc[model, f'{d}_{k}'] = v
            
    result, std = result.astype(float).round(3), std.astype(float).round(3)
    print(result)
    return result, std


# # Compare backbone concat for only RNA and both modalities 
# averages, stds = compare_model_architectures(data=['Pancreas'])
# dir_path = "results/qr/unimoda_rerun/Pancreas/"
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)
#     print("Directory created successfully!")
# else:
#     print("Directory already exists!")
# averages.to_csv(f"{dir_path}avg.csv")
# averages.to_csv(f"{dir_path}std.csv")

# Compare backbone concat for only RNA and both modalities 
averages, stds = compare_model_architectures(data=['ImmuneAtlas'])
dir_path = "results/qr/unimoda_rerun/ImmuneAtlas/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print("Directory created successfully!")
else:
    print("Directory already exists!")
averages.to_csv(f"{dir_path}avg.csv")
averages.to_csv(f"{dir_path}std.csv")

