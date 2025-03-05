import pandas as pd
import numpy  as np 
import pickle
import streamlit as st 
from streamlit_option_menu import option_menu
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif
# streamlit part
st.set_page_config(layout="wide")
st.title(":violet[BREAST CANCER DISEASE PREDICTION]")
selected = option_menu(
    menu_title=None,
    options= ["ABOUT", "APPLICATION"],
    menu_icon=None,
    icons=None,
    orientation="horizontal"
    #styles={}
)
# About a project description using streamlit
if selected=="ABOUT":
    st.header("PROJECT DESCRIPTION")
    st.text(" Breast cancer project focuses on predict whether the patient has cancer or not.")
    st.text("NumPy: A library for numerical computations in Python.")
    st.text("Pandas: A library for data manipulation and analysis.")
    st.text("Scikit-learn: A machine learning library that provides various regression and classification algorithms.")
    st.text('Matplotlib: A plotting library for creating visualizations.')
    st.text("Seaborn: A data visualization library built on top of Matplotlib.")

# To predict whether the patient has cancer or not
if selected=="APPLICATION":
    col1,col2=st.columns(2)
    with col1:
        texture_mean=st.number_input(":blue[ENTER TEXTURE_MEAN](Min 9.710 & Max 39.280)")
        perimeter_mean=st.number_input(":blue[ENTER PERIMETER_MEAN](Min 43.790 & Max 188.50)")
        smoothness_mean=st.number_input(":blue[ENTER SMOOTHNESS_MEAN](Min 0.052630 & Max 0.163400)")
        compactness_mean=st.number_input(":blue[ENTER COMPACTNESS_MEAN](Min 0.019380 & Max 0.345400)")
        concavity_mean=st.number_input(":blue[ENTER CONCAVITY_MEAN](Min 0.000000 & Max 0.426800)")
        fractal_dimension_mean=st.number_input(":blue[ENTER FRACTAL_DIMENSION_MEAN](Min 0.049960 & Max 0.079750)")
        texture_se=st.number_input(":blue[ENTER TEXTURE_SE](Min 0.360200 & Max 2.454150)")
    with col2:
        perimeter_se=st.number_input(":blue[ENTER PERIMETER_SE](Min 0.757000 & Max 5.983500)")
        fractal_dimension_se=st.number_input(":blue[ENTER FRACTAL_DIMENSION_SE](Min 0.000895 & Max 0.008023)")
        texture_worst=st.number_input(":blue[ENTER TEXTURE_WORST](Min 12.020000 & Max 42.680000)")
        perimeter_worst=st.number_input(":blue[ENTER PERIMETER_WORST](Min 50.410000 & Max 187.335000)")
        smoothness_worst=st.number_input(":blue[ENTER SMOOTHNESS_WORST](Min 0.072500 & Max 0.200100)")
        compactness_worst=st.number_input(":blue[ENTER COMPACTNESS_WORST](Min 0.027290 & Max 0.646950)")
        concavity_worst=st.number_input(":blue[ENTER CONCAVITY_WORST](Min 0.000000 & Max 0.795500)")
        data = {
                "texture_mean": [texture_mean],
                "perimeter_mean": [perimeter_mean],
                "smoothness_mean": [smoothness_mean],
                "compactness_mean": [compactness_mean],
                "concavity_mean": [concavity_mean],
                "fractal_dimension_mean": [fractal_dimension_mean],
                "texture_se": [texture_se],
                "perimeter_se": [perimeter_se],
                "fractal_dimension_se": [fractal_dimension_se],
                "texture_worst": [texture_worst],
                "perimeter_worst": [perimeter_worst],
                "smoothness_worst": [smoothness_worst],
                "compactness_worst": [compactness_worst],
                "concavity_worst": [concavity_worst],
                }

        df=pd.DataFrame(data,index=[1])

        with open("c:/Users/ADMIN/Desktop/projects_coding/mainproject/classification_model.pkl","rb")as f3:
            classi=pickle.load(f3)
        with open("c:/Users/ADMIN/Desktop/projects_coding/mainproject/classification_scale.pkl","rb")as f4:
            classi_scale=pickle.load(f4)

        new_df=classi_scale.transform(df)
        y_p=classi.predict(new_df)

        button=st.button(":violet[PREDICT BREAST CANCER]")
        if button:
            if y_p==1:
                st.write(':red[MALIGNANT: patient has cancer]')
            else:
                st.balloons()
                st.write(':green[BANIGN: Patient has no cancer]')