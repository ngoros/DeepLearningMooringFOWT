import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('predictions.csv')
a = np.zeros((5,5))

for i in range(len(df)):
    vm_p, m_p, t_p, s_p, vs_p = 0, 0, 0, 0, 0
    vm_t, m_t, t_t, s_t, vs_t = 0, 0, 0, 0, 0
    if df['Biof_pred'][i] <= 0.2:
        vm_p = 1
        if df['Biof_true'][i] <= 0.2:
            vm_t=1
            a[0,0] += 1
        elif 0.2 < df['Biof_true'][i] <= 0.4:
            m_t=1
            a[0,1] += 1
        elif 0.4 < df['Biof_true'][i] <= 0.6:
            t_t=1
            a[0,2] += 1
        elif 0.6 < df['Biof_true'][i] <= 0.8:
            s_t=1
            a[0,3] += 1
        elif 0.8 < df['Biof_true'][i] <= 1.0:
            vs_t=1
            a[0,4] += 1
    if 0.2 < df['Biof_pred'][i] <= 0.4:
        m_p = 1
        if df['Biof_true'][i] <= 0.2:
            vm_t=1
            a[1,0] += 1
        elif 0.2 < df['Biof_true'][i] <= 0.4:
            m_t=1
            a[1,1] += 1
        elif 0.4 < df['Biof_true'][i] <= 0.6:
            t_t=1
            a[1,2] += 1
        elif 0.6 < df['Biof_true'][i] <= 0.8:
            s_t=1
            a[1,3] += 1
        elif 0.8 < df['Biof_true'][i] <= 1.0:
            vs_t=1
            a[1,4] += 1
    if 0.4 < df['Biof_pred'][i] <= 0.6:
        t_p = 1
        if df['Biof_true'][i] <= 0.2:
            vm_t=1
            a[2,0] += 1
        elif 0.2 < df['Biof_true'][i] <= 0.4:
            m_t=1
            a[2,1] += 1
        elif 0.4 < df['Biof_true'][i] <= 0.6:
            t_t=1
            a[2,2] += 1
        elif 0.6 < df['Biof_true'][i] <= 0.8:
            s_t=1
            a[2,3] += 1
        elif 0.8 < df['Biof_true'][i] <= 1.0:
            vs_t=1
            a[2,4] += 1
    if 0.6 < df['Biof_pred'][i] <= 0.8:
        s_p = 1
        if df['Biof_true'][i] <= 0.2:
            vm_t=1
            a[3,0] += 1
        elif 0.2 < df['Biof_true'][i] <= 0.4:
            m_t=1
            a[3,1] += 1
        elif 0.4 < df['Biof_true'][i] <= 0.6:
            t_t=1
            a[3,2] += 1
        elif 0.6 < df['Biof_true'][i] <= 0.8:
            s_t=1
            a[3,3] += 1
        elif 0.8 < df['Biof_true'][i] <= 1.0:
            vs_t=1
            a[3,4] += 1
    if 0.8 < df['Biof_pred'][i] <= 1.0:
        vs_p = 1
        if df['Biof_true'][i] <= 0.2:
            vm_t=1
            a[4,0] += 1
        elif 0.2 < df['Biof_true'][i] <= 0.4:
            m_t=1
            a[4,1] += 1
        elif 0.4 < df['Biof_true'][i] <= 0.6:
            t_t=1
            a[4,2] += 1
        elif 0.6 < df['Biof_true'][i] <= 0.8:
            s_t=1
            a[4,3] += 1
        elif 0.8 < df['Biof_true'][i] <= 1.0:
            vs_t=1
            a[4,4] += 1
        
    if s_t==m_p==1:# or s_t==vm_p or vm_t==vs_p:
        print(i)

plt.style.use('science')
plt.figure(figsize=(10,10))
#sns.heatmap(confusion_matrix(y_test,predictions),cmap='viridis',annot=True)

b = np.zeros((5,5))
for i in range(0, 5):
    for j in range(0, 5):
        b[i, j] = a[i, j]/np.sum(a)

ax = sns.heatmap(a, cmap='Blues', annot=True, fmt='.0f', annot_kws={'size':28,'weight':'black'}, cbar=False,
                xticklabels=['Very mild','Mild', 'Interm.', 'Severe', 'Very severe'],
                yticklabels=['Very mild','Mild', 'Interm.', 'Severe', 'Very severe'])

ax.set_yticklabels(ax.get_yticklabels(), rotation = 90, fontsize = 14, va="center")
ax.tick_params(labelsize=14)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=16)
for t in ax.texts: t.set_text(t.get_text() + "%")
plt.savefig('biofconfusion.pdf')

a_true = [a[0,:], a[1,:], a[2,:], a[3,:], a[4,:]]
a_pred = [a[:,0], a[:,1], a[:,2], a[:,3], a[:,4]]

#dataa = pd.DataFrame(a)
#dataa.to_csv('biofouling.csv')
#precision = [a[0,0]/np.sum(a[0,:]), a[1,1]/np.sum(a[1,:]), a[2,2]/np.sum(a[2,:]), a[3,3]/np.sum(a[3,:]), a[4,4]/np.sum(a[4,:])]
#recall = [a[0,0]/np.sum(a[:,0]), a[1,1]/np.sum(a[:,1]), a[2,2]/np.sum(a[:,2]), a[3,3]/np.sum(a[:,3]), a[4,4]/np.sum(a[:,4])]
#f1_score = np.zeros(len(precision))
#for i in range(0, len(precision)):
#    f1_score[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])

#print('Biofoulingprecision: ')
#print(precision)
#print('Biofouling recall: ')
#print(recall)
#print('Biofouling F1-Score: ')
#print(f1_score)

a = np.zeros((5,5))

for i in range(len(df)):
    if df['Anchor_pred'][i] <= 0.2:
        if df['Anchor_true'][i] <= 0.2:
            a[0,0] += 1
        elif 0.2 < df['Anchor_true'][i] <= 0.4:
            a[0,1] += 1
        elif 0.4 < df['Anchor_true'][i] <= 0.6:
            a[0,2] += 1
        elif 0.6 < df['Anchor_true'][i] <= 0.8:
            a[0,3] += 1
        elif 0.8 < df['Anchor_true'][i] <= 1.0:
            a[0,4] += 1
    if 0.2 < df['Anchor_pred'][i] <= 0.4:
        if df['Anchor_true'][i] <= 0.2:
            a[1,0] += 1
        elif 0.2 < df['Anchor_true'][i] <= 0.4:
            a[1,1] += 1
        elif 0.4 < df['Anchor_true'][i] <= 0.6:
            a[1,2] += 1
        elif 0.6 < df['Anchor_true'][i] <= 0.8:
            a[1,3] += 1
        elif 0.8 < df['Anchor_true'][i] <= 1.0:
            a[1,4] += 1
    if 0.4 < df['Anchor_pred'][i] <= 0.6:
        if df['Anchor_true'][i] <= 0.2:
            a[2,0] += 1
        elif 0.2 < df['Anchor_true'][i] <= 0.4:
            a[2,1] += 1
        elif 0.4 < df['Anchor_true'][i] <= 0.6:
            a[2,2] += 1
        elif 0.6 < df['Anchor_true'][i] <= 0.8:
            a[2,3] += 1
        elif 0.8 < df['Anchor_true'][i] <= 1.0:
            a[2,4] += 1
    if 0.6 < df['Anchor_pred'][i] <= 0.8:
        if df['Anchor_true'][i] <= 0.2:
            a[3,0] += 1
        elif 0.2 < df['Anchor_true'][i] <= 0.4:
            a[3,1] += 1
        elif 0.4 < df['Anchor_true'][i] <= 0.6:
            a[3,2] += 1
        elif 0.6 < df['Anchor_true'][i] <= 0.8:
            a[3,3] += 1
        elif 0.8 < df['Anchor_true'][i] <= 1.0:
            a[3,4] += 1
    if 0.8 < df['Anchor_pred'][i] <= 1.0:
        if df['Anchor_true'][i] <= 0.2:
            a[4,0] += 1
        elif 0.2 < df['Anchor_true'][i] <= 0.4:
            a[4,1] += 1
        elif 0.4 < df['Anchor_true'][i] <= 0.6:
            a[4,2] += 1
        elif 0.6 < df['Anchor_true'][i] <= 0.8:
            a[4,3] += 1
        elif 0.8 < df['Anchor_true'][i] <= 1.0:
            a[4,4] += 1

b = np.zeros((5,5))
for i in range(0, 5):
    for j in range(0, 5):
        b[i, j] = a[i, j]/np.sum(a)

plt.figure(figsize=(10,10))
ax = sns.heatmap(a, cmap='Reds', annot=True, fmt=".0f", annot_kws={'size':28,'weight':'black'}, cbar=False,
                xticklabels=['Very mild','Mild', 'Interm.', 'Severe', 'Very severe'],
                yticklabels=['Very mild','Mild', 'Interm.', 'Severe', 'Very severe'])
ax.set_yticklabels(ax.get_yticklabels(), rotation = 90, fontsize = 14, va="center")
ax.tick_params(labelsize=14)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=16)
for t in ax.texts: t.set_text(t.get_text() + "%")
plt.savefig('anchorconfusion.pdf')

a_true = [a[0,:], a[1,:], a[2,:], a[3,:], a[4,:]]
a_pred = [a[:,0], a[:,1], a[:,2], a[:,3], a[:,4]]

#precision = [a[0,0]/np.sum(a[0,:]), a[1,1]/np.sum(a[1,:]), a[2,2]/np.sum(a[2,:]), a[3,3]/np.sum(a[3,:]), a[4,4]/np.sum(a[4,:])]
#recall = [a[0,0]/np.sum(a[:,0]), a[1,1]/np.sum(a[:,1]), a[2,2]/np.sum(a[:,2]), a[3,3]/np.sum(a[:,3]), a[4,4]/np.sum(a[:,4])]
#f1_score = np.zeros(len(precision))
#for i in range(0, len(precision)):
#    f1_score[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])

#print('Anchoring precision: ')
#print(precision)
#print('Anchoring recall: ')
#print(recall)
#print('Anchoring F1-Score: ')
#print(f1_score)
