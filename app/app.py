import pandas as pd
import numpy as np
from data_preprocessing import data_cleaning
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit_extras
from streamlit_extras.metric_cards import style_metric_cards
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
style_metric_cards(border_left_color="#FF4500")
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

#Menu - Tabs===================================================================================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Home', 'Seasonality', 'Route', 'Vehicles', 'Time', 'Rating', 'Prediction'])

#Home Page=====================================================================================================================================
with tab1:
    st.download_button(
        label="Download CSV Project File",
        data=csv_data,
        file_name='Fastfood Data Analysis.csv',
        mime='text/csv',
    )

    with st.expander('Dataset Information'):
        st.dataframe(df)

#Seasonality=====================================================================================================================================
with tab2:
    col1, col2 = st.columns(2)
    col1.metric('Total of Deliveries', len(df['id'].unique()))
    col2.metric('Total of Deliverymen', len(df['delivery_person_id'].unique()))

    col3, col4 = st.columns(2)
    with col3:
        aux = df.groupby(['order_day_month'])['id'].count().reset_index()
        graph = px.bar( aux, x = 'order_day_month', y = 'id', color = 'order_day_month', text_auto = '0.2s', title = 'DELIVERY BY MONTH')
        st.plotly_chart(graph, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux)
    with col4:
        aux1 = df.groupby(['number_week'])['id'].count().reset_index()
        graph1 = px.line( aux1, x = 'number_week', y = 'id', title = 'DELIVERY BY WEEK')
        st.plotly_chart(graph1, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)
 
    col5, col6 = st.columns(2)
    with col5:
        aux2 = df.groupby(['order_day_week'])['id'].count().reset_index()
        graph2 = px.bar( aux2, x = 'order_day_week', y = 'id', color = 'order_day_week', text_auto = '0.2s', title = 'DELIVERY BY DAY OF WEEK')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)
    with col6:     
        aux3 = df.groupby(['day_period'])['id'].count().reset_index()      
        graph3 = px.pie(aux3, names = 'day_period', values = 'id', color = 'day_period', title = 'DELIVERY BY DAY PERIOD')
        st.plotly_chart(graph3, use_container_width=True)
        with st.expander('More Info'):
            st.dataframe(aux3)

    aux3 = df.groupby(['order_date'])['id'].count().reset_index()
    graph3 = px.bar( aux3, x = 'order_date', y = 'id', text_auto = '0.2s', title = 'TIMELINE OF DELIVERY')
    st.plotly_chart(graph3, use_container_width = True)
    with st.expander('More Info'):
        st.dataframe(aux3)


#ROUTE=====================================================================================================================================
with tab3:
    col1, col2, col3 = st.columns(3)
    col1.metric('Total of Deliveries', len(df['id'].unique()))
    col2.metric('Types of food delivered', len(df['type_of_order'].unique()))
    col3.metric('Different cities', len(df['city'].unique()))

    col3, col4 = st.columns(2)
    with col3:
        aux = df.groupby(['city'])['id'].count().reset_index()
        graph = px.bar( aux, x = 'city', y = 'id', color = 'city', text_auto = '0.2s', title = 'DELIVERY BY CITY')
        st.plotly_chart(graph, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux)
    with col4:
        aux1 = df.groupby(['city'])['time_taken(min)'].mean().reset_index()
        graph1 = px.pie(aux1, names = 'city', values = 'time_taken(min)', color = 'city', title = 'DELIVERY TIME BY CITY')
        st.plotly_chart(graph1, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)
        
    aux3 = df.groupby(['weather_conditions'])['id'].count().reset_index()
    graph3 = px.line( aux3, x = 'weather_conditions', y = 'id', title = 'DELIVERY BY WEATHER CONDITIONS')
    st.plotly_chart(graph3, use_container_width = True)
    with st.expander('More Info'):
        st.dataframe(aux3)

    col8, col9 = st.columns(2)
    with col8:
        aux = df.groupby(['road_traffic_density'])['id'].count().reset_index()
        graph = px.bar( aux, x = 'road_traffic_density', y = 'id', color = 'road_traffic_density', text_auto = '0.2s', title = 'DELIVERY BY TRAFFIC')
        st.plotly_chart(graph, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux)
    with col9:
        aux1 = df.groupby(['road_traffic_density'])['time_taken(min)'].mean().reset_index()
        graph1 = px.line( aux1, x = 'road_traffic_density', y = 'time_taken(min)',title = 'DELIVERY TIME BY TRAFFIC')
        st.plotly_chart(graph1, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)

#Vehicles=====================================================================================================================================
with tab4:
    col1, col2 = st.columns(2)
    col1.metric('Total of Deliveries', len(df['id'].unique()))
    col2.metric('Number of vehicles', len(df['type_of_vehicle'].unique()))

    cols1, cols2 = st.columns(2)
    with cols1:
        aux2 = df.groupby(['type_of_vehicle'])['id'].count().reset_index()
        graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'id', color = 'type_of_vehicle', text_auto = '0.2s', title = 'DELIVERY BY VEHICLE')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    with cols2:
        aux2 = df.groupby(['type_of_vehicle'])['time_taken(min)'].mean().reset_index()
        graph2 = px.line( aux2, x = 'type_of_vehicle', y = 'time_taken(min)', title = 'DELIVERY TIME BY VEHICLE')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    cols1, cols2 = st.columns(2)
    with cols1:
        aux2 = df.groupby(['type_of_vehicle', 'road_traffic_density'])['time_taken(min)'].mean().reset_index()
        graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'time_taken(min)', color = 'road_traffic_density', text_auto = '0.2s', title = 'DELIVERY TIME BY TRAFFIC')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    with cols2:
        aux2 = df.groupby(['type_of_vehicle', 'weather_conditions'])['time_taken(min)'].mean().reset_index()
        graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'time_taken(min)', color = 'weather_conditions', text_auto = '0.2s', title = 'DELIVERY TIME BY WEATHER CONDITIONS')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    col1, col2 = st.columns(2)   
    with col1:
        aux2 = df.groupby(['type_of_vehicle', 'order_day_month'])['id'].count().reset_index()
        graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'id', color = 'order_day_month', text_auto = '0.2s', title = 'VEHICLE DELIVERY BY MONTH')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)
    with col2:
        aux2 = df.groupby(['type_of_vehicle', 'day_period'])['id'].count().reset_index()
        graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'id', color = 'day_period', text_auto = '0.2s', title = 'VEHICLE DELIVERY BY DAY PERIOD')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)
    c1, c2 = st.columns(2)
    with c1:
        aux2 = df.groupby(['type_of_vehicle', 'order_day_week'])['id'].count().reset_index()
        graph2 = px.line( aux2, x = 'type_of_vehicle', y = 'id', color = 'order_day_week', title = 'VEHICLE DELIVERY BY DAY OF THE WEEK')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)
    with c2:
        aux2 = df.groupby(['type_of_vehicle', 'order_day_month'])['id'].count().reset_index()
        graph2 = px.line( aux2, x = 'type_of_vehicle', y = 'id', color = 'order_day_month', title = 'VEHICLE DELIVERY BY MONTH')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

#Time=====================================================================================================================================
with tab5:
    st.header('teste')


#Ratings=====================================================================================================================================
with tab6:
    st.header('teste')