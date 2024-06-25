#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier ,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import sqlite3
from sqlalchemy import text,create_engine

con1  =sqlite3.connect('hearts_DB.db')

# c =con1.cursor()
# df=c.execute("SELECT * from Heart_Disease")

df =pd.read_sql("SELECT * from Heart_Disease",con1)
# %%
ab=df.isna().sum()
# %%
df.info





#%%
import matplotlib.pyplot as plt


# count occurrences of each category for each value of target
counts = df.groupby(['age', 'target']).size().unstack()

# plot bar chart
counts.plot(kind='bar', stacked=True)
plt.show()




# %%
cat_df = df[['sex','cp','fbs','restecg','exang','slope','ca','thal','target']]
num_df =df[['age','trestbps','chol','thalach','oldpeak','target']]


# %%
# Calculate Q1, Q3 and IQR for each column to detect outliers (Hoaglin, Iglewicz & Tukey, 1986)
Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1

# Determine if there are any values in each column that are outliers (Hoaglin, Iglewicz & Tukey, 1986)
outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).sum()

print(outliers)


#%%
import matplotlib.pyplot as plt  # matplotlib library is used for data visualization (Hunter, 2007)
import seaborn as sns  # seaborn library is used for statistical data visualization (Waskom, 2021)

plt.figure(figsize=(12, 6))
sns.boxplot(data=num_df)
plt.title('Box plots of all Numerical columns')
plt.show()


#%%
# Mean imputation to all numerical feature outliers
num_df[outliers] = np.nan
num_df.fillna(num_df.mean(), inplace=True)





#%%

# loop through each categorical variable and plot stacked bar chart
fig, axs = plt.subplots(nrows=4, ncols=2,figsize=(10,10))
axs = axs.flatten()

for i, col in enumerate(cat_df.columns[:-1]):
    counts = cat_df.groupby([col, "target"]).size().unstack()
    counts.plot(kind='bar', stacked=False, ax=axs[i])
    title = "Distribution of " + col + " by Target"
    axs[i].set_title(title)

plt.tight_layout()
plt.show()


# %%

# Part1

import seaborn as sns
fig, axs = plt.subplots(ncols=2, figsize=(15,10))
axs = axs.flatten()

# Set titles and labels for each plot

axs[0].set(title='Distribution of trestps by Target', xlabel='Target', ylabel='Frequency' )
axs[1].set(title='Distribution of chol by Target', xlabel='Target', ylabel='Frequency')
sns.set_theme(style="white")
sns.boxplot(x="target", y="trestbps", data=num_df, ax=axs[0] , color='forestgreen' )
sns.boxplot(x="target", y="chol", data=num_df, ax=axs[1], color='forestgreen')

plt.subplots_adjust(wspace=.4,hspace=.6)
plt.show()


# Part2

fig, axs = plt.subplots(ncols= 3, figsize=(15,10))
axs = axs.flatten()

axs[0].set(title='Distribution of thalach by Target', xlabel='Target', ylabel='Frequency')
axs[1].set(title='Distribution of oldpeak by Target', xlabel='Targete', ylabel='Frequency')
axs[2].set(title='Distribution of age by Target', xlabel='Target', ylabel='Frequency')

sns.boxplot(x="target", y="thalach", data=num_df, ax=axs[0], color='forestgreen')
sns.boxplot(x="target", y="oldpeak", data=num_df, ax=axs[1], color='forestgreen')
sns.boxplot(x="target", y="age", data=num_df, ax=axs[2], color='forestgreen')
plt.subplots_adjust(wspace=.4,hspace=.6)

plt.show()

# %%
import pandas as pd

import datetime
from numpy import mean
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression , SGDClassifier 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report , ConfusionMatrixDisplay , RocCurveDisplay 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE , BorderlineSMOTE
from collections import Counter
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split
from sklearn import metrics

from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek ,SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

# %%
df2 = df[['target','age','trestbps','chol','thalach','oldpeak','sex','cp','fbs','restecg','exang','slope','ca','thal']].copy()

df2[['age','trestbps','chol','thalach','oldpeak']] = scaler.fit_transform(
              df2[['age','trestbps','chol','thalach','oldpeak']])

# %%
# Split Target from the features
y = df2.iloc[:,0]
x = df2.iloc[:,1:]
counter = Counter(y)
print(counter)
sns.countplot(x='target',data=df ,color='forestgreen')
plt.title('Target Variable Distribution' )
plt.show()

# Reasonable balanced set
#%%
# Split the data in to combination set & validation set
x_comb, x_val , y_comb , y_val = train_test_split(x,y,train_size=0.8 ,random_state=11)
smote =SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),random_state=11)
# fit and apply the transform
x_rus, y_rus = smote.fit_resample(x_comb, y_comb)

# summarize class distribution resamples
print(Counter(y_rus))
sns.countplot(x='target',data=pd.DataFrame(y_rus) ,color='forestgreen')
plt.title('Target Variable Distribution Resampled' )
plt.show()
#%%

#Split Training from Testing
x_tr_rus, x_te_rus , y_tr_rus , y_te_rus = train_test_split(x_rus,y_rus,train_size=0.7 ,random_state=11)


# %%
lr1 = LogisticRegression()
SGD1 = SGDClassifier()
gbc = GradientBoostingClassifier()


'''Model 1'''
'''Gradient Bossting Classifier'''


'''Hyper Parameter Tuning'''
# %%
print('Hyper Parameter Tuning GBC')

scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall':make_scorer(recall_score),
            'roc_auc':make_scorer(roc_auc_score),
}
n_estimators_gb =[int(x) for x in np.linspace(start=5, stop=1000 , num=2)]
learning_rate_gb=[0.09,0.3,0.5]
max_depth_gb= [1,3,7]
param_grid_gb={'n_estimators': n_estimators_gb,
                'learning_rate': learning_rate_gb,
                'max_depth': max_depth_gb,
                # 'min_samples_split': np.linspace(0.0586),
                'min_samples_split': [0.05,0.8,0.1],
                'min_samples_leaf': [0.05,0.08,0.1],
                'subsample':[1,3],
                'loss':['log_loss'],
                'criterion': ['friedman_mse'],
                }

"""Fitting GBC"""
mdl_gbc2=gbc.fit(x_tr_rus,y_tr_rus)
mdl_gb_cv =GridSearchCV(estimator = mdl_gbc2,param_grid= param_grid_gb, cv=5 , verbose=2 ,n_jobs =-1,scoring='roc_auc')
mdl_gb_cv.fit(x_tr_rus,y_tr_rus)
mdl_gb_cv.best_params_




'''Training GBC'''

# %%
print('Testing GBC')

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
gbc_yp5= mdl_gb_cv.predict(x_te_rus)
cm=confusion_matrix(y_te_rus, gbc_yp5)
print(classification_report(y_te_rus , gbc_yp5))
# scores4 = cross_val_score(mdl_gb_cv
#                           , x_rus, y_rus
#                           , scoring='roc_auc', cv=cv, n_jobs=-1)

# print('Border CGB Mean ROC AUC: %.3f' % mean(scores4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mdl_gb_cv.classes_)
disp.plot()
plt.show()


y_prediction_train = mdl_gb_cv.predict_proba(x_te_rus)[:,1]
fpr, tpr, thresholds = roc_curve(y_te_rus, y_prediction_train)
roc_auc = roc_auc_score(y_te_rus, y_prediction_train)
 
print('GBC: ROC AUC=%0.2f' % (roc_auc))
# scores4 = cross_val_score(mdl_gbc2
#                           , x_resampled, y_resampled
#                           , scoring='roc_auc', cv=cv, n_jobs=-1)

# print('Border CGB Mean ROC AUC: %.3f' % mean(scores4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
# roc curve for tpr = fpr
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

'''Validate GBC'''

# %%
print('Validate GBC')

gbc_yp_val= mdl_gb_cv.predict(x_val)
mdl_gb_cv.score(x_val,y_val)
cm=confusion_matrix(y_val, gbc_yp_val)
print(classification_report(y_val , gbc_yp_val))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mdl_gb_cv.classes_)
disp.plot()
plt.show()

y_prediction_val = mdl_gb_cv.predict_proba(x_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_val, y_prediction_val)
roc_auc = roc_auc_score(y_val, y_prediction_val)
 
print('GBC Validate: ROC AUC=%0.2f' % (roc_auc))

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
# roc curve for tpr = fpr
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


'''GBC Feature Importance'''

#%%
print('GBC Feature Importance')
importances = pd.DataFrame(data={
    'Attribute': x_te_rus.columns,
    'Importance': mdl_gbc2.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)

plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()










""" Model 2 """
''' Logistic Regression '''

''' Training '''

# %%
print('Training & Testing Logistic R')

mdl_lr=lr1.fit(x_tr_rus,y_tr_rus)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
lr_yp5= mdl_lr.predict(x_te_rus)
cm=confusion_matrix(y_te_rus, lr_yp5)
print(classification_report(y_te_rus , lr_yp5))

scores4 = cross_val_score(mdl_lr
                          , x_rus, y_rus
                          , scoring='roc_auc', cv=cv, n_jobs=-1)

print('Border GBC Mean ROC AUC: %.3f' % mean(scores4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mdl_lr.classes_)
disp.plot()
plt.show()

y_prediction_te_lr = mdl_lr.predict_proba(x_te_rus)[:,1]
fpr, tpr, thresholds = roc_curve(y_te_rus, y_prediction_te_lr)
roc_auc = roc_auc_score(y_te_rus, y_prediction_te_lr)
 
print('Logistic Regression Test: ROC AUC=%0.2f' % (roc_auc))

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


'''Validate Logistic model'''

# %%
print('Validate Logistic R')
lr_yp_val= mdl_lr.predict(x_val)
mdl_lr.score(x_val,y_val)
cm=confusion_matrix(y_val, lr_yp_val)
print(classification_report(y_val , lr_yp_val))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mdl_lr.classes_)
disp.plot()
plt.show()


y_prediction_val_lr = mdl_lr.predict_proba(x_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_val, y_prediction_val_lr)
roc_auc = roc_auc_score(y_val, y_prediction_val_lr)
 
print('Logistiv Regression Validate: ROC AUC=%0.2f' % (roc_auc))

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
# roc curve for tpr = fpr
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()



'''Decile Validation Logistic regression'''

# %%
print('Decile Validatio Logistic R', ['bold'])
predicted_lr =mdl_lr.predict_proba(x_val)[:,1]
Target_lr =mdl_lr.predict(x_val)

Predictions_lr= pd.DataFrame({'actual':y_val ,'predicted_prob':predicted_lr,'predicted':Target_lr})

Predictions_lr=Predictions_lr.sort_values(by='predicted_prob')
Predictions_lr['Decile'] = pd.qcut(Predictions_lr['predicted_prob'], 10, labels=[10,9,8,7,6,5,4,3,2,1])

grouped_data = Predictions_lr.groupby('Decile')['actual'].sum()
Predictions_lr.groupby('Decile')['actual'].sum()
# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(grouped_data.index, grouped_data.values)
plt.xlabel('Decile')
plt.ylabel('Sum of Actual Values')
plt.title('Sum of Actual Values by Decile (Validation Data)')
plt.show()















'''Model 3'''
'''Stochastic Gradient Descent'''

''' Training & Testing'''
# %%
print('Training & Testing', ['bold'])
from sklearn.calibration import CalibratedClassifierCV
SGD2 = CalibratedClassifierCV(SGD1)
mdl_sgd=SGD2.fit(x_tr_rus,y_tr_rus)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
sgd_yp5= mdl_sgd.predict(x_te_rus)
cm=confusion_matrix(y_te_rus, sgd_yp5)
print(classification_report(y_te_rus , sgd_yp5))
# scores4 = cross_val_score(mdl_gb_cv
#                           , x_rus, y_rus
#                           , scoring='roc_auc', cv=cv, n_jobs=-1)

# print('Border CGB Mean ROC AUC: %.3f' % mean(scores4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mdl_sgd.classes_)
disp.plot()
plt.show()

y_prediction_te_sgd = mdl_sgd.predict_proba(x_te_rus)[:,1]
fpr, tpr, thresholds = roc_curve(y_te_rus, y_prediction_te_sgd)
roc_auc = roc_auc_score(y_te_rus, y_prediction_te_sgd)
 
print('SGD Test: ROC AUC=%0.2f' % (roc_auc))

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


'''Validate SGD'''

# %%
from simple_colors import *
print('Validate SGD')


sgd_yp_val= mdl_sgd.predict(x_val)
mdl_sgd.score(x_val,y_val)
cm=confusion_matrix(y_val, sgd_yp_val)
print(classification_report(y_val , sgd_yp_val))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=mdl_sgd.classes_)
disp.plot()
plt.show()


y_prediction_val_sgd = mdl_sgd.predict_proba(x_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_val, y_prediction_val_sgd)
roc_auc = roc_auc_score(y_val, y_prediction_val_sgd)
 
print('SGD Validate: ROC AUC=%0.2f' % (roc_auc))

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)')
# roc curve for tpr = fpr
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


'''Decile Validation SGD'''

# %%
print('Decile Validatio SGD')
predicted_sgd =mdl_sgd.predict_proba(x_val)[:,1]
Target_sgd =mdl_sgd.predict(x_val)

Predictions_sgd= pd.DataFrame({'actual':y_val ,'predicted_prob':predicted_sgd,'predicted':Target_sgd})

Predictions_sgd=Predictions_lr.sort_values(by='predicted_prob')
Predictions_sgd['Decile'] = pd.qcut(Predictions_sgd['predicted_prob'], 10, labels=[10,9,8,7,6,5,4,3,2,1])

grouped_data_sgd = Predictions_sgd.groupby('Decile')['actual'].sum()
Predictions_sgd.groupby('Decile')['actual'].sum()
# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(grouped_data_sgd.index, grouped_data_sgd.values)
plt.xlabel('Decile')
plt.ylabel('Sum of Actual Values')
plt.title('Sum of Actual Values by Decile (Validation Data)')
plt.show()


# %%
import pickle
with open("model_LR.pkl", "wb") as f:
    pickle.dump(mdl_lr, f)
    
# %%
