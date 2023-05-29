#%%
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from tkintertable import TableCanvas
import joblib
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#from matplotlib import rc

#################
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
    df1 = pd.read_csv("res/Table S2.csv", encoding='cp1252')
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
    pca_out = joblib.load('res/pca0928.pkl')  
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
###########################

###########################
### The first page
def form1():
    global root
    root = tk.Tk()
    root.title('ZirconfO2')
    root.geometry('1000x600+300+300')
    
    def tonext():
        form2()
        root.destroy()
    
    f1 = tk.Frame(root, height=590, width=200, bd = 3, relief="ridge")
    f1.place(x = 5, y = 5)
    
    f2 = tk.Frame(root, height=590, width=785, bd = 3, relief="ridge")
    f2.place(x = 210, y = 5)
    
    b0 = tk.Label(f1, text = 'Introduction', 
                  width = 20, height = 2,
                  font = ("Arial", 10, 'bold'),)
    b0.place(x = 10, y = 20)
    
    b1 = tk.Button(f1, text = "Start your calculation", 
                   width = 20, height = 2,
                   command = tonext, font = ("Arial", 10,'bold'),
                   bg="white",)
    b1.place(x = 10, y = 60)
    
    b2 = tk.Label(f2,
                  text ='\nZirconfO2 is a user-friendly software for calculation magma oxgen fugcity (fO2) using zircon trace elements. This oxybarometer is constucted based on the machine learning algorithm of Extremely Randomized Trees.\n \n \nFor the method and reference, please check:\nZou, S., Brzozowski, M.J., Chen X., Xu, D. Machine-Learning Oxybarometer Developed Using Zircon Trace-element Chemistry and Its Applications.https://doi.org/XXX.\n \n \nThe Online webapp can be also found in:\nhttps://shaohaozou-fo2-webapp-7xqvo0.streamlit.app\n \n \nDeveloper: Dr. Shaohao Zou, East China University of Technology, Nanchang, China, 330013\nEmail address: shaohaozou@hotmail.com',
                  wraplength=760,
                  font = ("Arial", 12),
                  justify=tk.LEFT)
    b2.place(x = 10, y = 10)
    
    b3 = tk.Label(f2,
                  text = "Latest news: The calclulator was updated at 2023-05-23.",
                  font = ("Arial", 10),
                  justify=tk.RIGHT
                  )
    b3.place(x = 400, y = 560)
    

    photo = tk.PhotoImage(file = "res/organization.png")
    b4 = tk.Label(f2, 
                  image=photo,
                  justify=tk.CENTER)
    b4.place(x = 100, y = 310)

    root.mainloop()

#########################
## The second page
def form2():
    root.destroy()
    
    root2 = tk.Tk()
    root2.title('ZirconfO2')
    root2.geometry('1000x600+300+300')
    
    filename = tk.StringVar()
    
    f11 = tk.Frame(root2, height=590, width=200, bd = 3, relief="ridge")
    f11.place(x = 5, y = 5)
    
    f22 = tk.Frame(root2, height=590, width=785, bd = 3, relief="ridge")
    f22.place(x = 210, y = 5)
    
    def file_selection():
        global data1
        ##select a CSV file
        file = filedialog.askopenfile(mode ='r', filetypes =[('csv', '*.csv')])
        data1 = pd.read_csv(file)
        data = data1.T.to_dict()
        table = TableCanvas(f22, data = data)
        table.show()
        table.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=tk.YES)
        #return data1
        
    def calculation():
        global df_result
        ##Get the dataset use for ML calculation
        df_ML = ML_dataset(data1)
        ### Machine learning calculation
        df_ML1 = df_ML.iloc[:, -7:]
        
        ## download the trained machine learning model. <if the model revised here need to be changed>
        
        y_predicted = joblib.load('res/best_ETsTotal.pkl').predict(df_ML1)
        y_pred = pd.DataFrame({"Pred_fO2(dFMQ)": y_predicted})
        df_label = df_ML.iloc[:, 1:7]
        ## Obtain the ML calculate result
        df_result = pd.concat([df_label.reset_index(), y_pred], axis = 1)
        df_result = df_result[['Sample Name', 'Analysis ID', 'Pred_fO2(dFMQ)']]
        
        return df_result
    
    ## asksaveasfile confirm the file name
    def saveresult():
        df_result.to_csv(filedialog.asksaveasfile(defaultextension='.csv'))
        
    ## Draw the result
    def plot():
        f3 = tk.Frame(root2, height=590, width=785, bd = 2, relief="ridge")
        f3.place(x = 210, y = 5)
        f3.destroy()
        f3 = tk.Frame(root2, height=600, width=800, bd = 2, relief="ridge")
        f3.place(x = 250, y = 30)
        
        
        fig, ax = plt.subplots(1,1, figsize=(6, 5),dpi = 100)
        
        sns.histplot(data = df_result, x = 'Pred_fO2(dFMQ)', binwidth = 0.15, kde = False, edgecolor='k', facecolor = 'aquamarine')
        ax.axvline(x = df_result['Pred_fO2(dFMQ)'].mean(), linestyle='--', 
                   color='r', linewidth=1,
                   label = 'Recommended value' + ' = ' + '{:.2f}'.format(df_result['Pred_fO2(dFMQ)'].mean()) + '±' + '{:.2f}'.format(df_result['Pred_fO2(dFMQ)'].std())
                   )
        ax.axvspan(np.mean(df_result['Pred_fO2(dFMQ)'])-np.std(df_result['Pred_fO2(dFMQ)']), np.mean(df_result['Pred_fO2(dFMQ)'])+np.std(df_result['Pred_fO2(dFMQ)']), facecolor='k', alpha = 0.2)
        
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), prop = {'size':10})
        ax.set_xlabel('f$\mathregular{O_2}$ (∆FMQ)',fontsize = 12)
        ax.set_ylabel('Count',fontsize = 12)
        
        canvas_spice = FigureCanvasTkAgg(fig, master = f3)
        canvas_spice.draw()
        canvas_spice.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=tk.YES)
        
        toolbar = NavigationToolbar2Tk(canvas_spice, f3)
        toolbar.update()
        canvas_spice._tkcanvas.pack(side=tk.TOP,fill=tk.BOTH,expand=tk.YES)

    
    ###The left position#
    
    b0 = tk.Label(f11, text = "Data input",
                  width = 20, height = 1,
                  font = ("Arial", 10,'bold'),
                  )
    b0.place(x = 10, y = 20)
    
    b21 = tk.Label(f22,
                  text ='\nStart your calculation accroding to the following steps:\n \n \nStep 1: Organize your data with the format that same as template CSV file, which can be found in the appendix of this paper.\n \nStep 2: Click the "Choose a CSV file" button to upload your dataset.\n \nStep 3: Click the "Calculate" button to finish the calculation.\n \nStep 4: Click the "Save result" button to download your result. \n \nIf you want to plot your result, you can also use the "Plot" button to review your result and get the recommend (average) value of your sample.\n \n \nIf you have any question, please contact Shaohao Zou for further help. \nEmail address: shaohaozou@hotmail.com',
                  wraplength=760,
                  font = ("Arial", 12),
                  justify=tk.LEFT)
    b21.place(x = 10, y = 10)
    
    
    b1 = tk.Button(f11, text = "Choose a CSV file", 
                   width = 22, height = 2,
                   command = file_selection, 
                   font = ("Arial", 10,'bold'),
                   bg="white",)
    b1.place(x = 6, y = 50)
    
    b2 = tk.Button(f11, text = "Calculation", 
                   width = 22, height = 2,
                   command = calculation, 
                   font = ("Arial", 10,'bold'),
                   bg="white",anchor=tk.CENTER)
    b2.place(x = 6, y = 110)
    
    b3 = tk.Button(f11, text = "Save result", 
                   width = 22, height = 2,
                   command = saveresult, 
                   font = ("Arial", 10,'bold'),
                   bg="white",anchor=tk.CENTER)
    b3.place(x = 6, y = 150)
    
    b4 = tk.Button(f11, text = "Plot", 
                   width = 22, height = 2,
                   command = plot,
                   font = ("Arial", 10,'bold'),
                   bg="white",anchor=tk.CENTER)
    b4.place(x = 6, y = 190)

    root2.mainloop()

if __name__=='__main__':
    form1()

# %%