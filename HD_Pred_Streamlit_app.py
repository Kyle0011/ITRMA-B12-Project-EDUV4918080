#%%
import streamlit as st
import time
import os 

import base64
import json
# from fuzzywuzzy import process, fuzz
import pyodbc
import pandas as pd
import numpy as np
import altair as alt

import streamlit as st
import matplotlib.pyplot as plt2
import subprocess
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

import sqlite3
from sqlalchemy import text,create_engine
from PIL import Image

# Load an image
image = Image.open('TVD1.png')
image2 = Image.open('docs.jpg')
imageHH = Image.open('Healthy.png')
imageHD = Image.open('HD.jpg')


# Display the image

# %%
con1  =sqlite3.connect('hearts_DB.db')
c =con1.cursor()

# %%
df =pd.read_sql("SELECT * from Heart_Disease",con1)

counts = df.groupby(['age', 'target']).size().unstack()

# %%
cat_df = df[['sex','cp','fbs','restecg','exang','slope','ca','thal','target']]
num_df =df[['age','trestbps','chol','thalach','oldpeak','target']]
# %%

# create subplots for each categorical variable
fig1, axs = plt.subplots(nrows=4, ncols=2,figsize=(10,10))
axs = axs.flatten()

for i, col in enumerate(cat_df.columns[:-1]):
    counts = cat_df.groupby([col, "target"]).size().unstack()
    counts.plot(kind='bar', stacked=False, ax=axs[i])
    title = "Distribution of " + col + " by Target"
    axs[i].set_title(title)

plt.tight_layout()
# st.pyplot(fig1)
# %%
# Part1

import seaborn as sns
fig2, axs = plt.subplots(ncols=2, figsize=(15,10))
axs = axs.flatten()

# Set titles and labels for each plot

axs[0].set(title='Distribution of trestps by Target', xlabel='Target', ylabel='Frequency' )
axs[1].set(title='Distribution of chol by Target', xlabel='Target', ylabel='Frequency')
sns.set_theme(style="white")
sns.boxplot(x="target", y="trestbps", data=num_df, ax=axs[0] , color='forestgreen' )
sns.boxplot(x="target", y="chol", data=num_df, ax=axs[1], color='forestgreen')

plt.subplots_adjust(wspace=.4,hspace=.6)



# Part2

fig3, axs = plt.subplots(ncols= 3, figsize=(15,10))
axs = axs.flatten()

axs[0].set(title='Distribution of thalach by Target', xlabel='Target', ylabel='Frequency')
axs[1].set(title='Distribution of oldpeak by Target', xlabel='Targete', ylabel='Frequency')
axs[2].set(title='Distribution of age by Target', xlabel='Target', ylabel='Frequency')

sns.boxplot(x="target", y="thalach", data=num_df, ax=axs[0], color='forestgreen')
sns.boxplot(x="target", y="oldpeak", data=num_df, ax=axs[1], color='forestgreen')
sns.boxplot(x="target", y="age", data=num_df, ax=axs[2], color='forestgreen')
plt.subplots_adjust(wspace=.4,hspace=.6)


with open("model_LR.pkl", "rb") as f:
    loaded_model = pickle.load(f)


# with open("HD_Prediction.py", "r") as fl:
#   exec(fl.read())

# df.columns
# Execute another Python script
# subprocess.call(['python', 'C:/Users/se5014/Documents/Studies/ITDAA/Project/HD_Prediction.py'])
# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark") 
st.title('💔 Heart Disease Predictor')
#col1, col2 = st.columns(2)
tab1,  tab3 = st.tabs(["Live Prediction", "Ditribution Charts"])

CP_dict_ = {'Chest Pain Type': '[Code]'
                ,"Typical Angina": [0] 
                ,'Atypical Angina': [1] 
                 ,"Non-anginal Pain": [2] 
                ,'Asymptomatic': [3] 

                } 

cp_list = list(CP_dict_.keys())[1:]

Sex_dict_ = {'sex': '[Code]'
                ,'Male': [1] 
                ,'Female': [0] 

                } 
sex_list = list(Sex_dict_.keys())[1:]

TV_dict_ = {'Code': '[TV]'
                ,'[1]': ['Heart Disease'] 
                ,'[0]': ['Healthy'] 

                } 
TV_list = list(TV_dict_.keys())[1:]


exang_dict_ = {'exang': '[Code]'
                ,"Yes": [1] 
                ,'No': [0] 

                } 
exang_list = list(exang_dict_.keys())[1:]


fbs_dict_ = {'Fasting Blooad Sugar': '[Code]'
                ,"True": [1] 
                ,'False': [0] 

                } 
fbs_list = list(fbs_dict_.keys())[1:]


restecg_dict_ = {'restecg': '[Code]'
                ,"Normal": [0] 
                ,'Abnormal': [1] 
                 ,"Ventricular Hypertrophy": [2] 


                } 

restecg_list = list(restecg_dict_.keys())[1:]

slope_dict_ = {'slope': '[Code]'
                ,"Upsloping": [0] 
                ,'Flat': [1] 
                 ,"Downsloping": [2] 


                } 

slope_list = list(slope_dict_.keys())[1:]

ca_dict_ = {'ca': '[Code]'
                ,"Mild": [0] 
                ,'Moderate': [1] 
                 ,"Severe": [2] 


                } 

ca_list = list(ca_dict_.keys())[1:]

thal_dict_ = {'thal': '[Code]'
                ,"Normal": [1] 
                ,'Fixed defect': [2] 
                 ,"Reversible Defect": [3] 


                } 

thal_list = list(thal_dict_.keys())[1:]

def cp_selector():
    cp_selection = st.sidebar.selectbox("Patient's chest pain type",(cp_list))
    cp_selection = cp_selection
    return cp_selection

def slope_selector():
    slope_selection = st.sidebar.selectbox("The slope of the peak ST segment",(slope_list))
    slope_selection = slope_selection
    return slope_selection

def ca_selector():
    ca_selection = st.sidebar.selectbox("Number of major vessels colored by fluoroscopy",(ca_list))
    ca_selection = ca_selection
    return ca_selection

def exang_selector():
    exang_selection = col_3.selectbox("Exercise induced angina",(exang_list))
    exang_selection = exang_selection
    return exang_selection

def restecg_selector():
    restecg_selection = st.sidebar.selectbox("Resting electrocardigraphic results",(restecg_list))
    restecg_selection = restecg_selection
    return restecg_selection

def thal_selector():
    thal_selection = col_4.selectbox("Status of the patient's heart",(thal_list))
    thal_selection = thal_selection
    return thal_selection


def fbs_selector():
    fbs_selection = st.sidebar.selectbox("Does the patient have high fasting blood sugar (>120 mg/dl)",(fbs_list))
    fbs_selection = fbs_selection
    return fbs_selection



def sex_selector():
    sex_selection = col_2.selectbox("Patient's sex",(sex_list))
    sex_selection = sex_selection
    return sex_selection


def Age_selector():
    #additional_excess = st.sidebar.number_input('Voluntary Excess (R)', value=0 ,step=1, placeholder = 'Specify how much additional excess you would like to pay...')
    age_sb = col_1.number_input("Patient's age: ", 1,150, 1)
    age_sb = age_sb
    return age_sb

def HR_selector():
    #additional_excess = st.sidebar.number_input('Voluntary Excess (R)', value=0 ,step=1, placeholder = 'Specify how much additional excess you would like to pay...')
    HR_sb = col_3.number_input('Maximum heart rate achieved: ',min_value= 1,max_value=200, step=15)
    HR_sb = HR_sb
    return HR_sb


def chol_selector():
    #additional_excess = st.sidebar.number_input('Voluntary Excess (R)', value=0 ,step=1, placeholder = 'Specify how much additional excess you would like to pay...')
    chol_sb = col_4.number_input('Serum cholesterol (mg/dl): ', 1,500, 1)
    chol_sb = chol_sb
    return chol_sb

def oldpeak_selector():
    #additional_excess = st.sidebar.number_input('Voluntary Excess (R)', value=0 ,step=1, placeholder = 'Specify how much additional excess you would like to pay...')
    oldpeak_sb = col_3.number_input('ST depression induced relative to rest: ', 1,100, 1)
    oldpeak_sb = oldpeak_sb
    return oldpeak_sb

def trestps_selector():
    #additional_excess = st.sidebar.number_input('Voluntary Excess (R)', value=0 ,step=1, placeholder = 'Specify how much additional excess you would like to pay...')
    trestps_sb = col_4.number_input('Resting blood pressure: ', 1,100, 1)
    trestps_sb = trestps_sb
    return trestps_sb



# Sidebar
with st.sidebar:
    
    # st.title('💔 Heart Disease Predictor')
    st.markdown('#### 💔 Please enter/select the following fields and then press the predict button!')
    
    cp_sb_selection = cp_selector()
    col_1,col_2 = st.sidebar.columns(2)
    sex_sb_selection = sex_selector()
    age_sb_selection = Age_selector()
    restecg_sb_selection= restecg_selector()
    col_3,col_4 = st.sidebar.columns(2)
    exang_sb_selection=exang_selector()
    thal_sb_selection = thal_selector()

    thalach_sb_selection = HR_selector()
    
    chol_sb_selection = chol_selector()
    oldpeak_sb_selection =oldpeak_selector()
    trestps_sb_selection = trestps_selector()
    fbs_sb_selection = fbs_selector()
    ca_sb_selection = ca_selector()
    slope_sb_selection = slope_selector()
    Predict_button = st.sidebar.button('Predict') 
    
chart_dict_ = {'Select a chart': '[Code]'
               ,"Target Variable Distribution by Class": [3] 
                ,"Categorical Variable Distribution by Target": [1] 
                ,'Numerical Variables Dustribution by Target': [2] 
                 


                } 

chart_list = list(chart_dict_.keys())[1:]

def chart_selector():
    chart_selection = tab3.selectbox("### Select a chart ###",(chart_list))
    chart_selection = chart_selection
    return chart_selection

with tab3:
    chart_selection = chart_selector()
    if  chart_selection=="Categorical Variable Distribution by Target":
        
        tab3.write('''### **Categorical** Variables Distribution Charts per Target Variable Class:''')
        tab3.write('###')
        tab3.pyplot(fig1)
    elif chart_selection=="Numerical Variables Dustribution by Target":
        
        tab3.write('''### Trestps & chol Variables Distribution Charts per Target Variable Class:''')
        tab3.write('###')
        tab3.pyplot(fig2)
        tab3.write('###')
        tab3.write('''### Thalalch, oldpeak & age Variables Distribution Charts per Target Variable Class:''')
        tab3.write('###')
        tab3.pyplot(fig3)
        tab3.write('###')
    else:
        tab3.write('''### Target Variable Distribution Chart per Class:''')
        tab3.write('###')
        tab3.image(image, caption='', use_column_width=True)


    
if Predict_button:
    progress_text = "Fetching data & Predicting..."
    percent_complete = 80
    my_bar = tab1.progress(percent_complete, text=progress_text)
    time.sleep(3)  
    my_bar.empty()
    progress_text = "Done Predicting..."
    percent_complete = 100
    my_bar = tab1.progress(percent_complete, text=progress_text)
    time.sleep(2) 
    my_bar.empty()
    tab1.write('### The following details have been entered for Patient X:')
    tab1.write('###')

    df_view =pd.DataFrame(data={
                                'Patient' : ['XXXX']
                                ,'age': [age_sb_selection]
                                , 'sex':[sex_sb_selection]
                                , 'cp':[cp_sb_selection]
                                , 'trestbps':[trestps_sb_selection]
                                , 'chol':[chol_sb_selection]
                                , 'fbs' :[fbs_sb_selection]
                                ,'restecg' :[restecg_sb_selection]
                                ,'thalach' :[thalach_sb_selection]
                                , 'exang' :[exang_sb_selection]
                                , 'oldpeak' :[oldpeak_sb_selection]
                                , 'slope' :[slope_sb_selection]
                                , 'ca' :[ca_sb_selection]
                                , 'thal' : [thal_sb_selection]} )
    tab1.dataframe(df_view)
    df_p =pd.DataFrame(data={
                                # 'Patient' : ['XXXX']
                                'age': [age_sb_selection]
                                , 'trestbps':[trestps_sb_selection]
                                , 'chol':[chol_sb_selection]
                                ,'thalach' :[thalach_sb_selection]
                                , 'oldpeak' :[oldpeak_sb_selection]
                                , 'sex':[Sex_dict_[str(sex_sb_selection)][0]]
                                , 'cp':[CP_dict_[str(cp_sb_selection)][0]]
                                
                                
                                , 'fbs' :[fbs_dict_[str(fbs_sb_selection)][0]]
                                ,'restecg' :[restecg_dict_[str(restecg_sb_selection)][0]]
                                
                                , 'exang' :[exang_dict_[str(exang_sb_selection)][0]]
                                
                                , 'slope' :[slope_dict_[str(slope_sb_selection)][0]]
                                , 'ca' :[ca_dict_[str(ca_sb_selection)][0]]
                                , 'thal' : [thal_dict_[str(thal_sb_selection)][0]]
                                }
                       )
    
    
    
    x_e = df_p

    lr_prediction= loaded_model.predict(x_e)
    lr_propa = loaded_model.predict_proba(x_e)[:,1]
    a =lr_prediction[0]
    lr_propa_percentage = str(round(lr_propa[0] * 100,2))+' %'
    if a == 0:
           Pred_print = '#### Therefore we predict the patient is fortunetely ' + TV_dict_[str(lr_prediction)][0]
    else:
           Pred_print = '#### Therefore we predict the patient unfortunetlely has ' + TV_dict_[str(lr_prediction)][0]

    tab1.write('###')
    tab1.markdown('''#### According to our trusted **Logistic Regression model**, the propability that Paient X has Heart Disease is: ''')
    if round(lr_propa[0] * 100,2) >=50:
        tab1.markdown(f"<h1 style='text-align: center; color: red;'>{lr_propa_percentage}</h1>", unsafe_allow_html=True)
        tab1.markdown(Pred_print) 
        tab1.image(imageHD, caption='', use_column_width=True)  
        
    else:
        tab1.markdown(f"<h1 style='text-align: center; color: green;'>{lr_propa_percentage}</h1>", unsafe_allow_html=True)
        tab1.markdown(Pred_print)
        tab1.image(imageHH, caption='', use_column_width=True)  
else: 
    progress_text = "Waiting on Patient X's details..."
    percent_complete = 0
    
    tab1.markdown(f"<h1 style='text-align: center; color: red;'>{'Please Press the button on the sidebar to generate a prediction!!'}</h1>", unsafe_allow_html=True)
    tab1.image(image2, caption='', use_column_width=True)   
    my_bar = tab1.progress(percent_complete, text=progress_text)
    
    # sgd_prediction= loaded_model.predict_proba(x_e)
    # loaded_model.score(x_val,y_val)
    # cm=confusion_matrix(y_val, sgd_yp_val)
    # print(classification_report(y_val , sgd_yp_val))
    # col1,col2,col3 = st.columns(3)






    # ab= col1.number_input('NONO',min_value=100,value=2000)
    # ab3= col1.multiselect('NONO',options=['a','b'])



# chol_sb_selection2 = chol_selector2()
# Get input from user
# additional_excess = get_add_excess(age_sb_selection) 

    
# tab1.title('OWO') 






    
# %%
