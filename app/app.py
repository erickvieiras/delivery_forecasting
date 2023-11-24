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
import xgboost as xgb
import pickle

# Page config and load data===================================================================================================================
st.set_page_config(layout="wide")
style_metric_cards(border_left_color="#FF4500")
df = pd.read_csv('../dataset/train.csv', low_memory= False)
df = data_cleaning(df)

# Load Pickle model============================================================================================================================

pickle_load = open('../model/xgb_model.pkl', 'rb')
prediction = pickle.load(pickle_load)

# Function to ML Model Forecasting

def forecasting(city, day_period, day_week, traffic, weather, vehicle, festival, lat, lon):
    
    prediction_result = prediction.predict([[city, day_period, day_week, traffic, weather, vehicle, festival, lat, lon]])
    print(prediction_result)
    return prediction_result

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Home', 'Seasonality', 'Route', 'Vehicles', 'Time', 'Rating', 'Forecasting'])

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
        graph = px.bar( aux, x = 'order_day_month', y = 'id', color = 'order_day_month', text_auto = '0.2s', title = 'TOTAL DELIVERY BY MONTH')
        st.plotly_chart(graph, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux)
    with col4:
        aux1 = df.groupby(['number_week'])['id'].count().reset_index()
        graph1 = px.line( aux1, x = 'number_week', y = 'id', title = 'TOTAL DELIVERY BY WEEK')
        st.plotly_chart(graph1, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)
 
    col5, col6 = st.columns(2)
    with col5:
        aux2 = df.groupby(['order_day_week'])['id'].count().reset_index()
        graph2 = px.bar( aux2, x = 'order_day_week', y = 'id', color = 'order_day_week', text_auto = '0.2s', title = 'TOTAL DELIVERY BY DAY OF WEEK')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)
    with col6:     
        aux3 = df.groupby(['day_period'])['id'].count().reset_index()      
        graph3 = px.pie(aux3, names = 'day_period', values = 'id', hole = 0.5, color = 'day_period', title = 'TOTAL DELIVERY BY DAY PERIOD')

        st.plotly_chart(graph3, use_container_width=True)
        with st.expander('More Info'):
            st.dataframe(aux3)

    aux3 = df.groupby(['order_date'])['id'].count().reset_index()
    graph3 = px.bar( aux3, x = 'order_date', y = 'id', text_auto = '0.2s', title = 'TOTAL TIMELINE OF DELIVERY')
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
        graph = px.bar( aux, x = 'city', y = 'id', color = 'city', text_auto = '0.2s', title = 'TOTAL DELIVERY BY CITY')
        st.plotly_chart(graph, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux)
    with col4:
        aux1 = df.groupby(['city'])['time_taken(min)'].mean().reset_index()
        graph1 = px.pie(aux1, names = 'city', values = 'time_taken(min)', hole = 0.5, color = 'city', title = 'AVERAGE DELIVERY TIME BY CITY')
        st.plotly_chart(graph1, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)
        
    aux3 = df.groupby(['weather_conditions'])['id'].count().reset_index()
    graph3 = px.line( aux3, x = 'weather_conditions', y = 'id', title = 'TOTAL DELIVERY BY WEATHER CONDITIONS')
    st.plotly_chart(graph3, use_container_width = True)
    with st.expander('More Info'):
        st.dataframe(aux3)

    col8, col9 = st.columns(2)
    with col8:
        aux = df.groupby(['road_traffic_density'])['id'].count().reset_index()
        graph = px.bar( aux, x = 'road_traffic_density', y = 'id', color = 'road_traffic_density', text_auto = '0.2s', title = 'TOTAL DELIVERY BY TRAFFIC')
        st.plotly_chart(graph, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux)
    with col9:
        aux1 = df.groupby(['road_traffic_density'])['time_taken(min)'].mean().reset_index()
        graph1 = px.line( aux1, x = 'road_traffic_density', y = 'time_taken(min)',title = 'AVERAGE DELIVERY TIME BY TRAFFIC')
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
        graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'id', color = 'type_of_vehicle', text_auto = '0.2s', title = 'TOTAL DELIVERY BY VEHICLE')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    with cols2:
        aux2 = df.groupby(['type_of_vehicle'])['time_taken(min)'].mean().reset_index()
        graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'time_taken(min)', color = 'type_of_vehicle', text_auto = '0.2s', title = 'AVERAGE DELIVERY TIME BY VEHICLE')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    cols1, cols2 = st.columns(2)
    with cols1:
        operation = st.selectbox('Select the Operation: ', ('Traffic', 'Weather'))
        if operation == 'Traffic':
            aux2 = df.groupby(['type_of_vehicle', 'road_traffic_density'])['time_taken(min)'].mean().reset_index()
            graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'time_taken(min)', color = 'road_traffic_density', text_auto = '0.2s', title = 'AVERAGE DELIVERY TIME BY TRAFFIC')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)
        else:
            aux2 = df.groupby(['type_of_vehicle', 'weather_conditions'])['time_taken(min)'].mean().reset_index()
            graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'time_taken(min)', color = 'weather_conditions', text_auto = '0.2s', title = 'AVERAGE DELIVERY TIME BY WEATHER CONDITIONS')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)

    with cols2:
        operation2 = st.selectbox('Select the Operation: ', ('Month', 'Day Period'))
        if operation2 == 'Month':
            aux2 = df.groupby(['type_of_vehicle', 'order_day_month'])['id'].count().reset_index()
            graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'id', color = 'order_day_month', text_auto = '0.2s', title = 'TOTAL VEHICLE DELIVERY BY MONTH')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)
        else:
            aux2 = df.groupby(['type_of_vehicle', 'day_period'])['id'].count().reset_index()
            graph2 = px.bar( aux2, x = 'type_of_vehicle', y = 'id', color = 'day_period', text_auto = '0.2s', title = 'TOTAL VEHICLE DELIVERY BY DAY PERIOD')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)

    aux2 = df.groupby(['type_of_vehicle', 'order_day_month'])['id'].count().reset_index()
    graph2 = px.line( aux2, x = 'type_of_vehicle', y = 'id', color = 'order_day_month', title = 'TOTAL VEHICLE DELIVERY BY MONTH')
    st.plotly_chart(graph2, use_container_width = True)
    with st.expander('More Info'):
        st.dataframe(aux2)

#Time=====================================================================================================================================
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        operation4 = st.selectbox('Select the Operation:  ', ('Traffic', 'Weather'))
    with c2:
        bool_param = st.selectbox('Select Order: ', ('Lowest', 'Higher'))

    if bool_param == 'Higher':
        bool_param = False
    else:
        bool_param = True

    if operation4 == 'Traffic':
        aux1 = df.groupby(['type_of_vehicle', 'road_traffic_density'])['time_taken(min)'].mean().reset_index().sort_values(['time_taken(min)'], ascending = bool_param)
        graph1 = px.bar( aux1, x = 'type_of_vehicle', color = 'road_traffic_density', y = 'time_taken(min)', text_auto='0.2s', title = 'AVERAGE OF THE TOP VEHICLE DELIVERIES BY TYPE OF TRAFFIC')
        st.plotly_chart(graph1, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)

    else:
        aux1 = df.groupby(['type_of_vehicle', 'weather_conditions'])['time_taken(min)'].mean().reset_index().sort_values(['time_taken(min)'], ascending = bool_param)
        graph1 = px.bar( aux1, x = 'type_of_vehicle', color = 'weather_conditions', y = 'time_taken(min)', text_auto='0.2s', title = 'AVERAGE OF THE TOP VEHICLE DELIVERIES BY TYPE OF WEATHER')
        st.plotly_chart(graph1, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux1)


    operation3 = st.selectbox('Select the Operation: ', ('Month', 'Week', 'Day of Week', 'Day Period'))
    col1, col2 = st.columns(2)
    
    with col1:
        if operation3 == 'Month':
            aux2 = df.groupby(['order_day_month'])['time_taken(min)'].sum().reset_index()
            graph2 = px.pie( aux2, names = 'order_day_month', values = 'time_taken(min)', hole = 0.5, title = 'TOTAL DELIVERY TIME BY MONTH')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)

        elif operation3 == 'Week':
            aux2 = df.groupby(['number_week'])['time_taken(min)'].mean().reset_index()
            graph2 = px.line( aux2, x = 'number_week', y = 'time_taken(min)', title = 'AVERAGE DELIVERY TIME BY WEEK')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)

        elif operation3 == 'Day of Week':
            aux2 = df.groupby(['order_day_week'])['time_taken(min)'].mean().reset_index()
            graph2 = px.bar( aux2, x = 'order_day_week', y = 'time_taken(min)', color = 'order_day_week', text_auto = '0.2s', title = 'DELIVERY TIME BY DAY OF WEEK')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)
        else:
            aux2 = df.groupby(['day_period'])['time_taken(min)'].mean().reset_index()
            graph2 = px.bar( aux2, x = 'day_period', y = 'time_taken(min)', color = 'day_period', text_auto = '0.2s', title = 'DELIVERY TIME BY DAY PERIOD')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)        

    with col2:
        if operation3 == 'Month':
            aux2 = df.groupby(['road_traffic_density', 'order_day_month'])['time_taken(min)'].mean().reset_index()
            graph2 = px.bar( aux2, x = 'road_traffic_density', y = 'time_taken(min)', color = 'order_day_month', text_auto = '0.2s', title = 'TYPE OF DELIVERY TRAFFIC BY MONTH')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)

        elif operation3 == 'Week':
            aux2 = df.groupby(['road_traffic_density', 'number_week'])['time_taken(min)'].mean().reset_index()
            graph2 = px.line( aux2, x = 'road_traffic_density', y = 'time_taken(min)', color = 'number_week', title = 'TYPE OF DELIVERY TRAFFIC BY WEEK')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)

        elif operation3 == 'Day of Week':
            aux2 = df.groupby(['road_traffic_density', 'order_day_week'])['time_taken(min)'].mean().reset_index()
            graph2 = px.bar( aux2, x = 'road_traffic_density', y = 'time_taken(min)', color = 'order_day_week', text_auto = '0.2s', title = 'TYPE OF DELIVERY TRAFFIC BY DAY OF WEEK')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2)
        else:
            aux2 = df.groupby(['road_traffic_density', 'day_period'])['time_taken(min)'].mean().reset_index()
            graph2 = px.bar( aux2, x = 'road_traffic_density', y = 'time_taken(min)', color = 'day_period', text_auto = '0.2s', title = 'TYPE OF DELIVERY TRAFFIC BY DAY PERIOD')
            st.plotly_chart(graph2, use_container_width = True)
            with st.expander('More Info'):
                st.dataframe(aux2) 

    if operation3 == 'Month':
        aux2 = df.groupby(['weather_conditions', 'order_day_month'])['time_taken(min)'].mean().reset_index()
        graph2 = px.bar( aux2, x = 'weather_conditions', y = 'time_taken(min)', color = 'order_day_month', text_auto = '0.2s', title = 'TIME DELIVERY OF WEATHER CONDITIONS BY MONTH')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    elif operation3 == 'Week':
        aux2 = df.groupby(['weather_conditions', 'number_week'])['time_taken(min)'].mean().reset_index()
        graph2 = px.line( aux2, x = 'weather_conditions', y = 'time_taken(min)', color = 'number_week', title = 'TIME DELIVERY OF WEATHER CONDITIONS BY WEEK')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    elif operation3 == 'Day of Week':
        aux2 = df.groupby(['weather_conditions', 'order_day_week'])['time_taken(min)'].mean().reset_index()
        graph2 = px.bar( aux2, x = 'weather_conditions', y = 'time_taken(min)', color = 'order_day_week', text_auto = '0.2s', title = 'TIME DELIVERY OF WEATHER CONDITIONS BY DAY OF WEEK')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2)
    else:
        aux2 = df.groupby(['weather_conditions', 'day_period'])['time_taken(min)'].mean().reset_index()
        graph2 = px.bar( aux2, x = 'weather_conditions', y = 'time_taken(min)', color = 'day_period', text_auto = '0.2s', title = 'TIME DELIVERY OF WEATHER CONDITIONS BY  DAY PERIOD')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2) 

#Ratings=====================================================================================================================================
with tab6:


    cols1 , cols2 = st.columns(2)
    with cols1:

        aux2 = df.groupby(['time_range'])['delivery_person_ratings'].mean().reset_index().sort_values(['delivery_person_ratings'], ascending=False).drop_duplicates(subset=['delivery_person_ratings'])
        graph2 = px.bar(aux2, x='time_range', y='delivery_person_ratings', color='time_range', text_auto='0.2s', title='AVERAGE RATINGS BY DELIVERY RANGE')
        st.plotly_chart(graph2, use_container_width=True)
        with st.expander('More Info'):
            st.dataframe(aux2)

    with cols2:
        aux2 = df.groupby(['road_traffic_density'])['delivery_person_ratings'].mean().reset_index().sort_values(['delivery_person_ratings'], ascending=False).drop_duplicates(subset=['delivery_person_ratings'])
        graph2 = px.line( aux2, x = 'road_traffic_density', y = 'delivery_person_ratings', title = 'AVERAGE RATINGS BY TRAFFIC TYPE')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2) 

    columns1 , columns2 = st.columns(2)
    with columns1:
        aux2 = df.groupby(['weather_conditions'])['delivery_person_ratings'].mean().reset_index().sort_values(['delivery_person_ratings'], ascending=False).drop_duplicates(subset=['delivery_person_ratings']).head(10)
        graph2 = px.line( aux2, x = 'weather_conditions', y = 'delivery_person_ratings', title = 'AVERAGE RATINGS BY WEATHER')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2) 
    with columns2:
        aux2 = df.groupby(['city'])['delivery_person_ratings'].mean().reset_index()
        graph2 = px.pie( aux2, names = 'city', values = 'delivery_person_ratings', hole = 0.5, title = 'AVERAGE RATINGS BY TYPE OF CITY')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2) 

    aux2 = df.groupby(['order_day_month', 'ratings_level'])['delivery_person_ratings'].sum().reset_index()
    graph2 = px.bar( aux2, x = 'order_day_month', y = 'delivery_person_ratings', color = 'ratings_level', text_auto='0.2s', title = 'RATINGS BY MONTH')
    st.plotly_chart(graph2, use_container_width = True)
    with st.expander('More Info'):
        st.dataframe(aux2) 
    

    columns3 , columns4 = st.columns(2)
    with columns3:
        aux2 = df.groupby(['delivery_person_id'])['time_taken(min)'].mean().reset_index().sort_values(['time_taken(min)'], ascending = True).drop_duplicates(subset = ['delivery_person_id']).head(10)
        graph2 = px.bar( aux2, x = 'delivery_person_id', y = 'time_taken(min)', text_auto = '0.2s', title = 'AVERAGE OF THE TOP 10 FASTEST DELIVERIERS')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2) 
    with columns4:
        aux2 = df.groupby(['delivery_person_id', 'ratings_level'])['delivery_person_ratings'].mean().reset_index().sort_values(['delivery_person_ratings'], ascending = False).drop_duplicates(subset = ['delivery_person_id']).head(10)
        graph2 = px.bar( aux2, x = 'delivery_person_id', y = 'delivery_person_ratings', color = 'ratings_level', text_auto = '0.2s', title = 'AVERAGE OF THE TOP 10 DELIVERIERS WITH BEST RATING')
        st.plotly_chart(graph2, use_container_width = True)
        with st.expander('More Info'):
            st.dataframe(aux2) 
#Forecasting===================================================================================================================================
with tab7:
    columns10, columns11 = st.columns(2)
    with columns10:
        city = st.selectbox('Select Type of City: ', (df['city'].unique()))
        day_period = st.selectbox('Select Day Period: ', (df['day_period'].unique()))
        day_week = st.selectbox('Select Day of Week: ', (df['order_day_week'].unique()))
        traffic = st.selectbox('Select Day Period: ', (df['road_traffic_density'].unique()))
        weather = st.selectbox('Select Type of Weathe Conditions: ', (df['weather_conditions'].unique()))
    with columns11:
        vehicle = st.selectbox('Select a Type of Vehicle: ', (df['type_of_vehicle'].unique()))
        festival = st.selectbox('Select Festival: ', (df['festival'].unique()))
        lat = st.selectbox('Select Latitude: ', (df['delivery_location_latitude'].unique()))
        lon = st.selectbox('Select Longitude: ', (df['delivery_location_longitude'].unique()))
    st.button('Predict')

    