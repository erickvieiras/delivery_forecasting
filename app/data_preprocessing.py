import pandas as pd
import numpy as np
from datetime import datetime
import random
import inflection

def data_cleaning(df):

    old_columns = df.columns
    new_columns = old_columns.map(lambda x: x.lower())
    df.columns = new_columns

    # Removing space within texts
    df['id']                 = df['id'].str.replace(' ', '')
    df['delivery_person_id'] = df['delivery_person_id'].str.replace(' ', '')
    df['type_of_order']      = df['type_of_order'].str.replace(' ', '')
    df['type_of_vehicle']    = df['type_of_vehicle'].str.replace(' ', '')

    # Selecting features other than 'NaN '
    df_dpa = df['delivery_person_age'].loc[df['delivery_person_age'] != 'NaN ']

    # Converting to the native variable type and performing an average
    df_dpa = df_dpa.astype(np.int64).mean()
    df_dpa = np.round(df_dpa, 0)

    # Applying a lambda function to replace values ​​equal to 'NaN ', with the arithmetic mean of the delivery age
    df['delivery_person_age'] = df['delivery_person_age'].apply(lambda x: df_dpa if x == 'NaN ' else x)

    # Selecting features other than 'NaN '
    df_dpr = df['delivery_person_ratings'].loc[df['delivery_person_ratings'] != 'NaN ']
    # Converting to the native variable type and performing an average
    df_dpr = pd.to_numeric(df_dpr, errors='coerce').mean()
    df_dpr = np.round(df_dpr, 1)
    # Applying a lambda function to replace values ​​equal to 'NaN ', with the arithmetic mean of the delivery rate
    df['delivery_person_ratings'] = df['delivery_person_ratings'].apply(lambda x: df_dpr if x == 'NaN ' else x)

    # Selecting features other than 'NaN '
    df_to = df['time_orderd'].loc[df['time_orderd'] != 'NaN ']

    # Changing the variable type to native
    df_to = pd.to_datetime(df_to)
    df_to = df_to.dt.strftime('%H:%M')
    df['time_orderd'] = df['time_orderd'].apply(lambda x: '18:30' if x == 'NaN ' else x)

    # Replacing 'NaN ' with the value of the most repeated variable .mode()
    df['road_traffic_density'] = df['road_traffic_density'].apply(lambda x: 'Low' if x == 'NaN ' else x)
    df['road_traffic_density'] = df['road_traffic_density'].str.replace(' ', '')

    # Replacing 'NaN ' with the value of the most repeated variable .mode()
    df['multiple_deliveries'] = df['multiple_deliveries'].apply(lambda x: '1' if x == 'NaN ' else x)

    # Replacing 'NaN ' with the value of the most repeated variable .mode()
    df['festival'] = df['festival'].apply(lambda x: 'No' if x == 'NaN ' else x)

    # Removing space within texts
    df['festival'] = df['festival'].str.replace(' ', '')

    # Replacing 'NaN ' with the value of the most repeated variable .mode()
    df['city'] = df['city'].apply(lambda x: 'Metropolitian' if x == 'NaN ' else x)

    # Removing space within texts
    df['city'] = df['city'].str.replace(' ', '')

    df['delivery_person_age']     = df['delivery_person_age'].astype(np.int64)
    df['delivery_person_ratings'] = df['delivery_person_ratings'].astype(float)
    df['order_date']              = pd.to_datetime(df['order_date'])
    df['time_orderd']             = pd.to_datetime(df['time_orderd'])
    df['time_order_picked']       = pd.to_datetime(df['time_order_picked'])
    df['vehicle_condition']       = df['vehicle_condition'].astype(np.int64)
    df['multiple_deliveries']     = df['multiple_deliveries'].astype(np.int64)

    # Removing the characters '(min) ', to convert the variable to an integer.
    df['time_taken(min)']         = df['time_taken(min)'].str.replace('(min) ', '')
    df['time_taken(min)']         = df['time_taken(min)'].astype(np.int64)

    # Converting hexadecimal ID column of type Object to base 16 int
    df['id'] = df['id'].apply(lambda x: int(x, 16))


    #FEATURE ENGINEERING

    #Calculating the order preparation time by subtracting the time the ticket is issued from the preparation time
    df['time_preparation(min)'] = (df['time_order_picked'] - df['time_orderd']).dt.total_seconds() / 60
    df['time_preparation(min)'] = df['time_preparation(min)'].apply(lambda x: np.abs(x) if x <= 0 else x)
    df['time_preparation(min)'] = df['time_preparation(min)'].astype(np.int64)

    # Extracting the time from the 'Time Orderd' attribute
    df['time_orderd'] = df['time_orderd'].dt.strftime('%H:%M')
    df['time_order_picked'] = df['time_order_picked'].dt.strftime('%H:%M')


    #Creating a new feature with the period of the day, based on the time intervals in which orders are issued
    df['day_period'] = df['time_orderd'].apply(lambda x: 'Morning' if (x >= '00:00') and (x <= '11:59') else 'Evening' if (x >= '12:00') and (x <= '17:59') else 'Night')

    # Removing the term 'Conditions' from the attribute 'Weather Conditions', because the attribute already indicates that the value refers to a condition. And standardizing the column with snakecase.
    df.rename({'weatherconditions': 'weather_conditions'}, inplace = True, axis = 1)
    df['weather_conditions'] = df['weather_conditions'].str.replace('conditions ', '')

    # Replacing the ' NaN' value with the one that most repeats: 'Fog'
    df['weather_conditions'] = df['weather_conditions'].str.replace(' ', '')
    df['weather_conditions'] = df['weather_conditions'].apply(lambda x: 'Fog' if x == 'NaN' else x)

    # Replacing the 'Jam' value for 'Extreme' of the 'Rad Traffic Density' attribute, for a better understanding of the variable.
    df['road_traffic_density'].replace({'Jam':'Extreme'}, inplace = True)

    # Extracting the day of the week and the month from the 'Order Date' attribute and creating a new Feature.
    df['order_day_week'] = df['order_date'].dt.day_name()
    df['order_day_month'] = df['order_date'].dt.month_name()

    df['order_day_week'] = df['order_day_week'].replace({'Sunday':'1 - Sunday','Monday': '2 - Monday', 'Tuesday': '3 - Tuesday' , 'Wednesday': '4 - Wednesday',
                                                       'Thursday': '5 - Thursday', 'Friday': '6 - Friday', 'Saturday': '7 - Saturday'})
    
    #Extracting the week number
    df['number_week']  = df['order_date'].apply(lambda data: data.strftime('%U'))
    df['number_week']  = df['number_week'].astype(np.int64)

    # Creating a new Feature based on the variables of the 'Delivery Person ratings' attribute, to help measure the transition of scale between assessments.
    df['ratings_level'] = df['delivery_person_ratings'].apply(lambda x: 'Excellent' if x >= 5 else 'Good' if (x <= 4.9) and (x > 4) else 'Medium' if (x <= 3.9) and (x > 2.9) else 'Bad')

    # Creating a delivery time range
    bins = [1, 10, 20, 25, 30, 35, 40, 45, 50, 55]
    df['time_range'] = pd.cut(df['time_taken(min)'], bins = bins) 
    df['time_range'] = df['time_range'].astype(str)

    # Rearranging the columns
    new_order = ['id', 'delivery_person_id', 'delivery_person_age', 'delivery_person_ratings', 'ratings_level', 
                'restaurant_latitude', 'restaurant_longitude', 'delivery_location_latitude', 'delivery_location_longitude', 
                'order_date', 'number_week', 'order_day_week', 'order_day_month', 'day_period', 'time_orderd', 'time_order_picked', 'time_preparation(min)',
                'weather_conditions', 'road_traffic_density', 'vehicle_condition', 'type_of_order', 'type_of_vehicle', 'multiple_deliveries', 
                'festival', 'city', 'time_range', 'time_taken(min)']

    df = df[new_order]

    return df