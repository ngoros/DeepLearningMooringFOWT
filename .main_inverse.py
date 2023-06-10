import os
from src.preprocessing import read_data
from src.training import run_training
from src.arch import *
from src.postprocessing import *
#from src.categories import confusion_matrices
from time import time

#archs_f = [arch11_f, arch12_f, arch13_f, arch14_f, arch15_f, arch16_f, arch17_f, arch18_f, arch19_f, arch20_f, arch21_f, arch22_f, arch23_f, arch24_f, arch25_f, arch26_f, arch27_f, arch28_f, arch29_f, arch30_f, arch31_f, arch32_f, arch33_f, arch34_f]

archs_f = [arch31_f]
archs_i = [arch3_i]

with open('network_scores.txt', 'w') as f:
    f.close()

for arch in archs_f:
    
    data_path = os.path.join('data', 'joint.csv')
    u_train, u_validation, r_train, r_validation, P_train, P_validation = read_data(data_path, debug=False)

    t1 = time()
    t2 = time()
    train_forward = False
    
    for inv_arch in archs_i:
        t3 = time()
        model, model_inverse, a, arch_i = run_training(train_forward, u_train, u_validation, r_train, r_validation, P_train, P_validation, arch, inv_arch)
        t4 = time() 
        plot_losses(model, 'total', name=str(archs_i.index(inv_arch)))

        if not train_forward:
            prediction(model, u_validation, r_validation, arch, arch_i, t1, t2, t3, t4)
            make_predictions(model, model_inverse, u_validation, r_validation, P_validation)
            # confusion_matrices()


