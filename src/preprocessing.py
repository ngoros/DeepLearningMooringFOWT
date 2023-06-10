import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def read_data(data_path, debug=False):
    df = pd.read_csv(data_path, header=None)
    df.columns= ['Hs', 'Tp', 'V',
            'Mean-Surge', 'SD-Surge', 'F1-Surge', 'F2-Surge', 'M-Surge',
            'Mean-Sway', 'SD-Sway', 'F1-Sway', 'F2-Sway', 'M-Sway',
            'Mean-Heave', 'SD-Heave', 'F1-Heave', 'F2-Heave', 'M-Heave',
            'Mean-Roll', 'SD-Roll', 'F1-Roll', 'F2-Roll', 'M-Roll',
            'Mean-Pitch', 'SD-Pitch', 'F1-Pitch', 'F2-Pitch', 'M-Pitch',
            'Mean-Yaw', 'SD-Yaw', 'F1-Yaw', 'F2-Yaw', 'M-Yaw',
            'C1', 'C2']

    X = df.drop(['C1', 'C2', 'F1-Heave', 'F2-Roll'], axis=1)
    P = df[['C1', 'C2']]
    indices = []

    #for i in range(0, len(X)):
    #    if X['Hs'][i] > 0.121*X['Tp'][i]**2:
    #        indices.append(i)

    #X = X.drop(indices)
    #P = P.drop(indices)

    X_train, X_test, P_train, P_test = train_test_split(X, P, test_size=0.10, random_state=101)

    u_train, u_test = X_train.drop(['Hs', 'Tp', 'V'], axis=1), X_test.drop(['Hs', 'Tp', 'V'], axis=1)
    r_train, r_test = X_train[['Hs', 'Tp', 'V']], X_test[['Hs', 'Tp', 'V']]
   
    print(r_train.shape) 
    u_train, u_test = scale_data(u_train, u_test)
    r_train, r_test = scale_data(r_train, r_test)
    
    if debug:
        print(u_train)
        print(P_train)
        print('u=', u_train.shape)
        print('r=', r_train.shape)
        print('P=', P_train.shape)

    return u_train, u_test, r_train, r_test, P_train, P_test

