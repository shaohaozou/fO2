#%%
from logging import warning
from pathlib import Path
##from tkinter import Button
import pandas as pd
import streamlit as st
import numpy as np

import datetime
import joblib

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


## Boostrap resampling method from Keller et al., 2012, nature, doi:10.1038/nature11024.

# data and sigma should be read  by pandas
def bsresample(data, sigma, nrows, p):
    resampled = np.zeros((nrows, data.shape[1]))
    #sdata = pd.DataFrame()
    # If we have more than one sample
    i = 0
    while i < nrows:
        # Select weighted sample of data
        if data.shape[0] >1:
            t = (np.random.rand(data.shape[0],1) < p)
            sdata = data[t].values
            
            if sigma.shape[0] >1:
                if pd.DataFrame(sigma.values).shape[1] == 1:
                    serr = sigma.to_frame()[t].values
                else:
                    serr = sigma[t].values
            else:
                serr = np.ones(sdata.shape) * sigma.values
                
        else: # If only one sample
            sdata = data.values
            serr = sigma.values
            
        # Randomize data over uncertainty interval
        sdata += np.random.randn(sdata.shape[0],sdata.shape[1])* serr

        if (i + sdata.shape[0]) <= nrows:
            resampled[i:(i+sdata.shape[0]), :] = sdata
        else:
            resampled[i:nrows,:] = sdata[0:(nrows-i),:]

            
        #state = sta
        i +=sdata.shape[0]
        
    return resampled

###############################################

################################
#def down_template():
### log_transfer the result, and 
def log_transfer(data):
    ### 1. La less than 0.1
    data = data.drop(data[data['La']>1].index)
    
    ### 2. P less than 2000
    data = data.drop(data[data['P']>2000].index)
    
    ### 3. Th/U between 0.2 to 4
    data['Th/U'] = data['Th'] / data['U']
    data = data.drop(data[data['Th/U'] < 0.1].index)
    data = data.drop(data[data['Th/U']>4].index)
    data = data.drop(['Th/U'], axis = 1)
    data = data.reset_index(drop = True)
    
    ele = ['P', 'Ti', 'Y', 'Nb', 'Hf', 'Th', 'U', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho','Er', 'Tm', 'Yb', 'Lu']
    
    df = data[ele]
    df[df <= 0] =np.nan
    
    ## Log transformation
    for i in range(len(ele)):
        df[ele[i]] = df[ele[i]].apply(np.log)
    
    df2 = pd.concat([data.drop(ele, axis =1),df], axis =1)
    return df2

### Construct the Machine learning dataset
def ML_dataset(data):
    ### 1. read the orignial data using for machine learning
    ### which is used for calculation the mean and the std log values (used for get the distribution)
    df1 = pd.read_csv("Table S2.csv", encoding='cp1252')
    df1 = log_transfer(df1)
    
    ### 1. log transfrom the calculated data
    data = log_transfer(data)
    ele = ['P', 'Ti', 'Y', 'Nb', 'Hf', 'Th', 'U', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho','Er', 'Tm', 'Yb', 'Lu']
    
    df =data[ele]
    #### filled the Nan values using normal distribution with mean and std.
    np.random.seed(1)
    for i in range(len(ele)):
        df[ele[i]][np.isnan(df[ele[i]])] = pd.DataFrame(np.random.normal(df1[ele[i]].mean(),df1[ele[i]].std(), len(df[ele[i]])))[0]
    
    ## if you want to add new data into the data set, the pca model should change.
    pca_out = joblib.load('pca0928.pkl')  
    pc_list = ["PC"+str(i) for i in list(range(1, 22))]
    pca_scores = pca_out.transform(df)
    indivaul_df = pd.DataFrame(pca_scores, columns=pc_list)
    
    ### chose sample number and PC1 to PC7
    df_ML = pd.concat([data.drop(ele, axis =1), indivaul_df.iloc[:, 0:7]], axis =1)
    
    return df_ML

### Obtain the Machine learning train and test X, Y; and also the testset location
def get_ML_XY(data,i = 0):
    X = data.iloc[:, -7:]
    y = data['log fO2 (dFMQ)']
    
    df_ML = pd.concat([data.iloc[:, 0:1], X, data[['log fO2 (dFMQ)', 'log fO2 (dFMQ)_1std']]], axis=1)
    
    Train, test = train_test_split(df_ML, test_size = 0.2, random_state = i, stratify = df_ML.iloc[:, 0:1])
    
    X_train = Train.iloc[:, 1:8]
    y_train = Train['log fO2 (dFMQ)']
    
    X_test = test.iloc[:, 1:8]
    y_test = test['log fO2 (dFMQ)']
    test_loc = test[[test.columns.values[0], 'log fO2 (dFMQ)', 'log fO2 (dFMQ)_1std']]
    
    return X_train, y_train, X_test, y_test, test_loc

### Machine learnin calculation

### Convert the result
def convert_df(df):
    return df.to_csv().encode("utf-8")

### Download template CSV file
def download_temp():
    data= pd.read_csv("template.csv", encoding='cp1252')
    
    st.markdown('---')
    st.markdown("""
        <h4 style = "text_align:center; font-weight: bold;"> Step 1: Add Your Data Into Template CSV File </h3>
        """, 
        unsafe_allow_html=True)
    
    st.download_button(label = "📥 Download the template CSV file",
                       data = convert_df(data), file_name = 'The template CSV file.csv')
    st.markdown('---')

### Read, calculate and display/download the result
def calculation_process():
    ## Read the result
    st.markdown("""
                <h4 style = "text_align:center; font-weight: bold;"> Step 2: Upload Your Orangnized Data (in template CSV format) </h3>
                """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file")
        
    if uploaded_file is not None:
        ### read the csv
        data = pd.read_csv(uploaded_file)
        
        with st.expander("Preview of the input data"):
            st.dataframe(data)
        #if st.button("Preview of the input data"):
            #st.dataframe(data)
        
        ## Get the dataset use for ML calculation
        df_ML = ML_dataset(data)
        
        ### Machine learning calculation
        df_ML1 = df_ML.iloc[:, -7:]
        ## download the trained machine learning model. <if the model revised here need to be changed>
        y_predicted = joblib.load('best_ETsTotal.pkl').predict(df_ML1)
        y_pred = pd.DataFrame({"Pred_fO2(dFMQ)": y_predicted})
        
        df_label = df_ML.iloc[:, 1:7]
        ## Obtain the ML calculate result
        df_result = pd.concat([df_label.reset_index(), y_pred], axis = 1)
        
        #st.warning('Result')
        ## download the result(the result format need to be reset)
        st.markdown('---')
        
        st.markdown("""
                    <h4 style = "text_align:center; font-weight: bold;"> Step 3: Calculation </h3>
                    """, unsafe_allow_html=True)
         
        if st.button('🔣 Calculate'):
            ## display the result
            st.dataframe(df_result)
            st.success('🎉Success! You can download your result below.')
            st.download_button(label = "📥 Download your results",
                               data = convert_df(df_result), file_name = 'your_result.csv')
            
    else:
        st.warning('Please upload your data')
        
    st.markdown('---')
    # return data

def main():
    st.set_page_config(layout="wide")
    
    st.markdown('Latest news: The calclulator was updated at 2023-05-23.')
    
    st.markdown("""
        <h2 style = "color:#2B60DE; text_align:center; font-weight: bold;"> Machine Learming Oxybarometer Using Trace Elements 
        of Zircons </h2>
        """,unsafe_allow_html=True)
    
    st.markdown("""
        <p style = "color:#0000A0; text_align:center; "> A machine learning-based oxybarometer developed by using trace elements of zircon. 
        Please upload your dataset and start the calculation! </p>
        """, unsafe_allow_html=True)
    
    
    st.markdown('---')

    st.markdown("""
        * **Developer:** Dr. **Shaohao Zou**; State Key Laboratory of Nuclear Resources and Environment, East China University of Technology, Nanchang, China
        * **Email:** <shaohaozou@hotmail.com>
        """,
        unsafe_allow_html=True)
    
    st.markdown('---')
    
    with st.expander('Click here for more info on how to use this Webapp ✨'):
        st.write(Path('README.md').read_text())
    
    download_temp()
    calculation_process()
    
    st.markdown("""
        #### Citation
        """, 
        unsafe_allow_html=True)
    st.markdown("""
                * Zou, S., Brzozowski, M.J., Chen X., Xu, D. Machine-Learning Oxybarometer Developed Using Zircon Trace-element Chemistry and Its Applications.https://doi.org/XXX
                """)

    with st.expander('Contacts'):
        with st.form(key = 'contact', clear_on_submit = True):
            
            email = st.text_input('Your contact Email')
            st.text_area("Query", "Please fill in all the information or we may not be able to process your request!")
            
            submit_button = st.form_submit_button(label = 'Send Information')
            

       
if __name__ == '__main__':
    main()
    


# %%
