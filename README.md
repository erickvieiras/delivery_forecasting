# Marketplace Delivery Analysis

Marketplace Delivery Analysis is a project created for a food delivery marketplace, in order to collect business metrics, evaluate the volume of data and the residual impact of elements that directly influence one or more deliveries. This data relates the restaurant, delivery person and customer based on strategic decision making.

Among the metrics that have a residual impact on the time spent on deliveries, we can highlight some such as:

1 - Vehicles and vehicle conditions
2 - Delivery date
3 - Delivery time
4 - Type of traffic density
5 - Weather conditions
6 - type of city
7 - classification of deliveries
8 - restaurant location
9 - delivery location
10 - order type
11 - Multiple deliveries
12 - age of delivery people

# Top Metrics:

Among some insights from the project, we can highlight:

1 - Seasonal deliveries;
2 - Deliveries by city, vehicle, type of traffic and weather conditions;
3 - Delivery time by city, vehicle, type of traffic and weather conditions;
4 - Relationship of time spent on delivery and classification between: Vehicles, weather conditions and type and traffic
5 - Ranking of delivery people
6 - Geolocation of the fastest deliveries
7 - Geolocation of the best rated restaurants.
8 - Algorithm making predictions based on the relationship between type and traffic, weather conditions, time of day, day of the week, vehicle and latitude and longitude.

# Top 10 Insights:

1 - The month of March saw the most deliveries with 78.5% more compared to the previous month. The month of February has 11.2% more deliveries compared to the month of April. Therefore, we can observe an 80% drop between the months of March and April. It also has more positive reviews compared to the months of February and April, with 57.6% above average.

2 - Deliveries tended to grow over the weeks. An increase of 34% compared from the 6th to the 7th week, however, closing deliveries in the last week with a drop of 12.7% compared to the 7th week, and 44% less compared to the 11th week with greater flow of registered deliveries.

3 - Wednesdays and Fridays are the days of the week with the highest flow of total deliveries, representing 11.4% for the other days.

4 - The night period has 63.9% more deliveries compared to the daytime periods combined.

5 - Metropolitan cities have a delivery flow of 71.3% more compared to Urban cities, and 99.5% more than Semi-Urban cities. And also with more reviews.

6 - Semi-Urban cities have a longer average delivery time compared to other types of cities.

7 - Low traffic has a higher delivery rate, and also a shorter average delivery time.

8 - Motorcycle is the vehicle that has the highest number of deliveries, representing 57.9% of total deliveries recorded, but it is also the vehicle that has the highest average delivery time with 26.7% compared to the others.

9 - The greatest time spent traveling for deliveries is in extreme traffic in all delivery months, while medium and high traffic have the same time spent.

10 - There is a downward trend in delivery classifications due to traffic density and weather conditions, the greater and worse the conditions, the lower the classification given by the customer. Sunny days and traffic have better ratings.


# Final product

The purpose of the project is to create an analytical Dashboard with a set of metrics not yet defined by the company, to assist in decision making, strategic business analysis, company assets and predict delivery time based on features. The Realtime project can be accessed through the Dashboard using the following link: https://delivery-analysis.streamlit.app/

# Considerations

The public database explored in this project did not have the original purpose of providing predictions, so it was necessary to adjust data and consider a median metric to evaluate the regression model used.
