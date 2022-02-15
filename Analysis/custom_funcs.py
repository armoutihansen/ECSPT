import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit

def split(data, strat_column='sid', random_state=181):
    data[['self_z','other_z']] = data[['self_z','other_z']].fillna(0)
    train, test = train_test_split(data, train_size=0.7, stratify=data[strat_column], random_state=random_state, shuffle=True)
    temp, cv_fold_1 = train_test_split(train, test_size=int(train.shape[0]*.2), stratify=train['sid'], random_state=181, shuffle=True)
    temp1, cv_fold_2 = train_test_split(temp, test_size=int(train.shape[0]*.2), stratify=temp['sid'], random_state=181, shuffle=True)
    temp2, cv_fold_3 = train_test_split(temp1, test_size=int(train.shape[0]*.2), stratify=temp1['sid'], random_state=181, shuffle=True)
    temp3, cv_fold_4 = train_test_split(temp2, test_size=int(train.shape[0]*.2), stratify=temp2['sid'], random_state=181, shuffle=True)
    cv_fold_5 = temp3
    
    train =  pd.get_dummies(train, columns=['sid'])
    test = pd.get_dummies(test, columns=['sid'])
    cv_fold_1 =  pd.get_dummies(cv_fold_1, columns=['sid'])
    cv_fold_2 =  pd.get_dummies(cv_fold_2, columns=['sid'])
    cv_fold_3 =  pd.get_dummies(cv_fold_3, columns=['sid'])
    cv_fold_4 =  pd.get_dummies(cv_fold_4, columns=['sid'])
    cv_fold_5 =  pd.get_dummies(cv_fold_5, columns=['sid'])
    
    iterations = [
    [pd.concat([cv_fold_1,cv_fold_2,cv_fold_3,cv_fold_4]), cv_fold_5],
    [pd.concat([cv_fold_1,cv_fold_2,cv_fold_3,cv_fold_5]), cv_fold_4],
    [pd.concat([cv_fold_1,cv_fold_2,cv_fold_4,cv_fold_5]), cv_fold_3],
    [pd.concat([cv_fold_1,cv_fold_3,cv_fold_4,cv_fold_5]), cv_fold_2],
    [pd.concat([cv_fold_2,cv_fold_3,cv_fold_4,cv_fold_5]), cv_fold_1],
               ]
    
    def set_test_fold(row):
        if row.idx in iterations[0][1].index.to_list():
            return 0
        elif row.idx in iterations[1][1].index.to_list():
            return 1
        elif row.idx in iterations[2][1].index.to_list():
            return 2
        elif row.idx in iterations[3][1].index.to_list():
            return 3
        else:
            return 4
        
    train['idx'] = train.index
    
    test_fold = train.apply(set_test_fold, axis=1)
    
    train = train.drop(columns=['idx'])
    
    ps = PredefinedSplit(test_fold)
    
    return (train, test, iterations, ps)