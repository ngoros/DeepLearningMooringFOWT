import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('predictions.csv')

matsize = 50
mat = np.zeros((matsize, matsize))
df['Anchor_pred'] = np.round(df['Anchor_pred'], 2)

factor = 100/(mat.shape[0]-1)

for i in range(len(df)):
    row = int(df['Anchor_pred'][i]*100/factor)
    column = int(df['Anchor_true'][i]*100/factor)
    mat[row, column] += 1
   
#for i in range(0, matsize):
    #mat[:, i] = mat[:, i][::-1]
    #mat[i, :] = mat[i, :][::-1]

matb = mat[1:,1:]

mb = pd.DataFrame(matb)
mb.to_csv('matb.csv')

line = [[0, 1], [0, 1]]

figures=1

plt.style.use('science')

plt.figure(figsize=(3.5,3.5))
ax = sns.heatmap(matb, vmax=np.max(matb), cmap='bone_r', cbar=None)
ax.set_xticks(np.linspace(0,matsize-1,6))
ax.set_yticks(np.linspace(0,matsize-1,6))
plt.xticks(rotation=0, fontsize=16)
plt.yticks(rotation=0, fontsize=16)
ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1])
ax.set(xlabel='Ground Truth', ylabel='Prediction')
ax.annotate('$r^2=0.9916$', (3, 41),
            fontsize=16)
plt.xlabel('Ground Truth', fontsize=16)
plt.ylabel('Prediction', fontsize=16)
plt.plot(np.arange(50), np.arange(50), 'r--', linewidth=1)
plt.xlim(0, matsize*0.95)
plt.ylim(0, (matsize-1)*0.95)
#plt.gca().invert_yaxis()
plt.savefig('heatmap_anchor.pdf')

if not figures:
    sum_a, sum_b = 0, 0
    for i in range(1, len(matb)):
        for j in range(1, len(matb)):
            sum_a += (df[j][i]-df[i][j])**2
            sum_b += (df[j][i]-np.sum(df[j][:]))

    print('R2 equals: '+str(1-sum_a/sum_b))

    actual, pred = [], []
    for i in range(1, len(matb)):
        for j in range(1, len(matb)):
            actual.append(df[j][i])
            pred.append(df[i][j])

    corr_matrix = np.corrcoef(pred, actual)
    print(corr_matrix)
    corr = corr_matrix[0, 1]
    print(corr)
    R = corr**2

    print('R2 equals: '+str(R))
############################################################################



if figures:


    matsize = 50
    mat = np.zeros((matsize, matsize))
    df['Biof_pred'] = np.round(df['Biof_pred'], 2)

    factor = 100/(mat.shape[0]-1)

    for i in range(len(df)):
        row = int(df['Biof_pred'][i]*100/factor)
        column = int(df['Biof_true'][i]*100/factor)
        mat[row, column] += 1
        
    matb = mat[1:,1:]

#plt.style.use('science')

    plt.figure(figsize=(3.5,3.5))
    ax = sns.heatmap(matb, vmax=np.max(matb), cmap='bone_r', cbar=None)
    ax.set_xticks(np.linspace(0,matsize-1,6))
    ax.set_yticks(np.linspace(0,matsize-1,6))
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1])
    ax.set(xlabel='Ground Truth', ylabel='Prediction')
    ax.annotate('$r^2=0.9807$', (3, 41),
            fontsize=16)
    plt.plot(np.arange(50), np.arange(50)*0.95, 'b--', linewidth=1)
    plt.xlim(0, matsize*0.95)
    plt.ylim(0, (matsize-1)*0.95)
    plt.xlabel('Ground Truth', fontsize=16)
    plt.ylabel('Prediction', fontsize=16)
    plt.savefig('heatmap_biof.pdf')
