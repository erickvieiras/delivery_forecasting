import pandas as pd
import numpy as np
from data_preprocessing import data_cleaning
import plotly.express as px
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import plotly.graph_objects as go
from PIL import Image
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor

# Page config and load data===================================================================================================================
st.set_page_config(layout="wide")
df = pd.read_csv('../dataset/train.csv', low_memory= False)
df = data_cleaning(df)

#function convert to dataset download=========================================================================================================
def convert_df(dataset):
    csv_data = dataset.to_csv(index=False)
    return csv_data

csv_data = convert_df(df)

#sidebar=======================================================================================================================================
#image = Image.open('../img/logo.png')
#st.sidebar.image(image)
st.sidebar.title('Delivery Analytics')

#Home Page=====================================================================================================================================
st.download_button(
        label="Download CSV Project File",
        data=csv_data,
        file_name='Fastfood Data Analysis.csv',
        mime='text/csv',
    )

with st.expander('Dataset Information'):
    st.dataframe(df)

