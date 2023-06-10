import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def plot_losses(model, x, name=''):
    metrics = pd.DataFrame(model.history.history)
    metrics['loss-log']=np.log10(metrics['loss'])
    metrics['val-loss-log']=np.log10(metrics['val_loss'])
    
    if x=='forward':
        metrics.to_csv('metrics_forward.csv', index=False)
    elif x=='total':
        metrics.to_csv('metrics_total'+str(name)+'.csv', index=False)

    plt.style.use('science')
    plt.figure(figsize=(4,3))
    plt.plot(np.arange(len(metrics)), metrics['loss-log'], 'k', label='Training')
    plt.plot(np.arange(len(metrics)), metrics['val-loss-log'], 'r', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Log($L$)')
    plt.legend(loc='best')

    if x=='forward':
        plt.savefig('losses_forward.pdf')
    elif x=='inverse':
        plt.savefig('losses_inverse.pdf')
    elif x=='total':
        plt.savefig('losses_total'+str(name)+'.pdf')

def prediction(model, u_validation, r_validation, arch_fun_f, arch_fun_i, t1, t2, t3, t4, co=''):
    predictions = model.predict([u_validation, r_validation])
    mean_error = mean_absolute_error(u_validation, predictions)
    expl_variance = explained_variance_score(u_validation, predictions)
    r2 = r2_score(u_validation, predictions)
    t=(t4-t3)+(t2-t1)
    print('Explained variance: '+str(expl_variance)+', r2: '+str(r2))
    
    with open('network_scores'+str(co)+'.txt', 'a') as f:
        f.write('{}, {}, {}, {}, {}\n'.format(arch_fun_f, arch_fun_i, expl_variance, r2, t))
    #plt.style.use('science')
    #plt.figure(figsize=(4,3))
    #plt.plot(u_validation, u_validation,'r');
    #plt.scatter(u_validation, predictions, linewidth=0);
    #plt.annotate((u_validation, predictions),(0,1))
    #plt.xlabel('Ground Truth')
    #plt.ylabel('Predicted')
    #plt.savefig('crossplot.pdf')
    
def prediction_forward(model, P_validation, r_validation):
    predictions = model.predict([P_validation, r_validation])
    mean_error = mean_absolute_error(P_validation, predictions)
    expl_variance = explained_variance_score(P_validation, predictions)
    r2 = r2_score(P_validation, predictions)
    print('Explained variance: '+str(expl_variance)+', r2: '+str(r2))

def make_predictions(model, model_inverse, u_validation, r_validation, P_validation):
    u_pred = model.predict([u_validation, r_validation])
    P_pred = model_inverse.predict([u_validation, r_validation])
    da = pd.DataFrame(r_validation)
    db = pd.DataFrame(u_validation)
    dc = pd.DataFrame(P_pred)
    #df.columns= ['Hs', 'Tp', 'V',
    #        'Mean-Surge', 'SD-Surge', 'F1-Surge', 'F2-Surge', 'M-Surge',
    #        'Mean-Sway', 'SD-Sway', 'F1-Sway', 'F2-Sway', 'M-Sway',
    #        'Mean-Heave', 'SD-Heave', 'F1-Heave', 'F2-Heave', 'M-Heave',
    #        'Mean-Roll', 'SD-Roll', 'F1-Roll', 'F2-Roll', 'M-Roll',
    #        'Mean-Pitch', 'SD-Pitch', 'F1-Pitch', 'F2-Pitch', 'M-Pitch',
    #        'Mean-Yaw', 'SD-Yaw', 'F1-Yaw', 'F2-Yaw', 'M-Yaw',
    #        'Biof_pred', 'Anchor_pred']
    #df.columns = ['Biof_pred', 'Anchor_pred']
    P_validation.columns = ['Biof_true', 'Anchor_true']
    P_validation.reset_index()
    dc['Biof_true'] = P_validation['Biof_true'].values
    dc['Anchor_true'] = P_validation['Anchor_true'].values
    dc.to_csv('predictions.csv', index=None)
    da.to_csv('conds.csv', index=None)
    db.to_csv('modals.csv', index=None)
