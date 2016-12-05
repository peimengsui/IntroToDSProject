
# coding: utf-8

# In[3]:

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# In[5]:

train = pd.read_csv("self_train.csv",index_col=0)
X_test = pd.read_csv("self_test.csv",index_col=0)


# In[6]:

X_train = train.drop("loss",axis=1)
Y_train = train['loss']


# In[7]:

catFeatureslist = [x for x in train.columns[0:-1] if 'cat' in x]
for cf in catFeatureslist:
    le = LabelEncoder()
    le.fit(X_train[cf].unique())
    X_train[cf] = le.transform(X_train[cf])


# In[8]:

for cf in catFeatureslist:
    le = LabelEncoder()
    le.fit(X_test[cf].unique())
    X_test[cf] = le.transform(X_test[cf])


# In[9]:

xgb_model = xgb.XGBRegressor()

#when in doubt, use xgboost
# parameters = {'nthread':[1], #when use hyperthread, xgboost may become slower
#               'objective':['reg:linear'],
#               'learning_rate': [0.06], #so called `eta` value
#               'max_depth': [6],
#               'silent': [1],
#               'subsample': [0.7],
#               'colsample_bytree': [0.6],
#               'n_estimators': [100], #number of trees
#               'seed': [1337]}
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.06], #so called `eta` value
              'max_depth': [4,6,8,10],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.4,0.6,0.8,1],
              'n_estimators': [100], #number of trees
              'seed': [1337]}


# In[10]:

def _score_func(estimator, X, y):
    return mean_absolute_error(np.expm1(y), np.expm1(estimator.predict(X)))


# In[11]:

clf = GridSearchCV(xgb_model, parameters, 
                   cv=KFold(len(Y_train), n_folds = 5,shuffle = True), 
                   scoring=_score_func,
                   verbose=2, refit=True)

clf.fit(X_train, Y_train)


# In[12]:

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])


# In[13]:

score


# In[ ]:

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.expm1(y),np.expm1(yhat))


# In[15]:

param = best_parameters
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, train_size=0.9)
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_validation,Y_validation)
dtest = xgb.DMatrix(X_test)
evallist  = [(dvalid,'eval'), (dtrain,'train')]
bst = xgb.train(param,
                dtrain,
                50000,
                evallist,
                early_stopping_rounds=50,
                feval = xg_eval_mae)


bst.save_model('xgboostfinal.model')


# In[16]:

ypred = bst.predict(dtest)


# In[17]:

prediction = np.expm1(ypred)


# In[18]:

submit = pd.read_csv("sample_submission.csv")


# In[19]:

submit["loss"] = prediction
submit.to_csv("xgbsubsetcvtrial.csv",index=False)


