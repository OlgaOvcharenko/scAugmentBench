import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale

def read_data(path: str):
    return pd.read_csv(path, sep=",")

# BC
df = read_data(path='results/table1.csv')
datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method" and col.endswith("Total")])
methods = list(df["Method"])
df.set_index('Method', inplace=True)

# Find ranking and make
rank_bc = []
for i, d in enumerate(datasets):
    rank_bc.append(np.array([df.loc[m, f"{d} Total"] for m in methods]))
rank_bc = np.array(rank_bc).mean(axis=0)
tmp = np.array([[x, y]for y, x in zip(rank_bc, methods)])
methods_bc = list(tmp[:, 0])
rank_bc = list(tmp[:, 1])


#  QR
df = read_data(path='results/tableB1_B2.csv')
datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method" and col.endswith("MacroF1")])
df.set_index('Method', inplace=True)

# Find ranking and make
rank_qr = []
for i, d in enumerate(datasets[:-1]):
    rank_qr.append(np.array([df.loc[m, f"{d} MacroF1"] for m in methods]))

df = read_data(path='results/table_B3.csv')
datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method" and col.endswith("MacroF1")])
df.set_index('Method', inplace=True)

# Find ranking and make
for i, d in enumerate(datasets[:-2]):
    rank_qr.append(np.array([df.loc[m, f"{d} MacroF1"] for m in methods]))
    
rank_qr = np.array(rank_qr).mean(axis=0)
tmp = np.array([[x, y]for y, x in zip(rank_qr, methods)])
methods_qr = list(tmp[:, 0])
rank_qr = list(tmp[:, 1])


# MP
df = read_data(path='results/table_B4.csv')
datasets = list([" ".join(col.split(" ")[:-1]) for col in df.columns if col != "Method"])
df.set_index('Method', inplace=True)

# Find ranking and make
rank_mp = []
for i, d in enumerate(datasets):
    rank_mp.append(np.array([df.loc[m, f"{d} Pearson"] for m in methods]))
rank_mp = np.array(rank_mp).mean(axis=0)
tmp = np.array([[x, y]for y, x in zip(rank_mp, methods)])
methods_mp = list(tmp[:, 0])
rank_mp = list(tmp[:, 1])

# print(methods_bc, rank_bc)
# print(methods_qr, rank_qr)
# print(methods_mp, rank_mp)

final_rates = np.array([rank_bc, rank_qr, rank_mp])
final_rates = minmax_scale(final_rates, axis=1, )
print(final_rates)


# Plot
fig, ax = plt.subplots(1, 1, figsize=(23/3.6, 4), dpi=500)
im = ax.imshow(final_rates, cmap="coolwarm")

ax.set_xticks(np.arange(len(methods)), labels=methods, fontsize=16,)
ax.set_yticks(np.arange(3), labels=["Batch\nCorrection", "Query-to-Reference\nMapping", "Missing Modality\nPrediction"], fontsize=16)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=-40, ha="right",
            rotation_mode="anchor", fontsize=16,)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, ticks=[0, 1],)
cbar.ax.set_ylabel("Scaled Score", rotation=-90, va="bottom", fontsize=16, labelpad=-10)
cbar.ax.tick_params(labelsize=16)

ax.set_xlabel("Methods", fontsize=20, loc='center') 
ax.set_ylabel("Tasks", fontsize=20,) 
ax.set_title("Methods Comparison", fontsize=20)

ax.set_aspect(aspect=2)

plt.subplots_adjust(left=0.41, right=0.99, top=0.62, bottom=0.08, wspace=0.02, hspace=0.7)
plt.savefig("plots/table_comparison.pdf")
plt.savefig("plots/table_comparison.svg")
