import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split(data, strat_column='sid', random_state=181):
    data[['self_z','other_z']] = data[['self_z','other_z']].fillna(0)
    train, test = train_test_split(data, train_size=0.7, stratify=data[strat_column], random_state=random_state, shuffle=True)
    temp, cv_fold_1 = train_test_split(train, test_size=int(train.shape[0]*.2), stratify=train['sid'], random_state=181, shuffle=True)
temp1, cv_fold_2 = train_test_split(temp, test_size=int(train.shape[0]*.2), stratify=temp['sid'], random_state=181, shuffle=True)
temp2, cv_fold_3 = train_test_split(temp1, test_size=int(train.shape[0]*.2), stratify=temp1['sid'], random_state=181, shuffle=True)
temp3, cv_fold_4 = train_test_split(temp2, test_size=int(train.shape[0]*.2), stratify=temp2['sid'], random_state=181, shuffle=True)
cv_fold_5 = temp3