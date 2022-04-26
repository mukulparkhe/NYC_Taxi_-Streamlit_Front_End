import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

#st.markdown(
#    """
#   <style>
#    .main {
#    background-color: #F5F5F5;
#    }
#    <style>
#    """,
#    unsafe_allow_html=True
# )



@st.cache
def get_data(fileName):
    taxi_data = pd.read_csv('C:/Users/mukul/Desktop/Streamlit1/Taxi.csv')
    
    return taxi_data



with header:
    st.title('Welcome to my awesome streamlit web applicaiton!')
    st.text('In this project I look into the transactions of taxis in NYC')

with dataset:
    st.header('NYC 2013 Green Taxi Trip Data')
    st.text('I found this dataset on NYC Open Data (https://data.cityofnewyork.us/Transportation/2013-Green-Taxi-Trip-Data/ghpb-fpea)')
    
    taxi_data = get_data('C:/Users/mukul/Desktop/Taxi.csv')
    #st.write(taxi_data.head())
    
    st.subheader('Total Amount paid per ride')
    Total_amount_dist = pd.DataFrame(taxi_data['Total_amount'].value_counts()).head(50)
    st.bar_chart(Total_amount_dist)
    

with features:
    st.header('10 Reasons to use Streamlit')
    
    st.markdown('* **Streamlit offers:**...')
    st.markdown('* Streamlit offers...')
    st.markdown('* Streamlit offers...')
    st.markdown('* Streamlit offers...')
    st.markdown('* Streamlit offers...')
    st.markdown('* Streamlit offers...')
    st.markdown('* Streamlit offers...')
    st.markdown('* Streamlit offers...')
    st.markdown('* Streamlit offers...')
    st.markdown('* Streamlit offers...')
    

with model_training:
    st.header('Time to train the model')
    st.text('Here you will get to choose the hyperparameters of the model and see how the performance changes!')
    
    sel_col, disp_col = st.columns(2)
    
    max_depth = sel_col.slider('what should be the max depth of the model?', min_value = 10, max_value = 100, value = 20, step = 10)
    
    n_estimators = sel_col.selectbox('How many trees there should be?', options = [100,200,300,'No Limit'], index = 0)
    
    sel_col.text('Here is a list of features in the dataset')
    sel_col.write(taxi_data.columns)
    input_feature =  sel_col.text_input('Which feature should be used as the input feature?','Total_amount')
    
    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth = max_depth)
    else:
        regr = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
    
    X = taxi_data[[input_feature]]
    Y = taxi_data[['Trip_distance']]
    
    regr.fit(X,Y)
    prediction = regr.predict(Y)
    
    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(Y, prediction))
                   
    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(Y, prediction))

    disp_col.subheader('R-squared score of the model is:')
    disp_col.write(r2_score(Y, prediction))




                   
                   