
import streamlit as st
import datetime
import pandas as pd
import datapackage

st.write("This program will never be able to predict the future values exactly. Although, It might give an intuition towards the current and potential future trend of a specific stock.")

# Date
st.title("Date range")

min_date = datetime.datetime(2015, 1, 1)
max_date = datetime.date(2020, 1, 1)

a_date = st.date_input("Pick a date", (min_date, max_date))

start = a_date[0]
end = a_date[1]

#Companies
data_url = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'
package = datapackage.Package(data_url)
resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
st.write(data)

#User Company Choice
companies  = data["Symbol"]

# User Inputs
def get_userInput():
    st.sidebar.write("Uses selected range as a training set")
    trained_date = st.sidebar.slider("Training Range", 1, 100, 60)
    st.sidebar.write("Tries to calculate the future values based on selected range")
    predict_day = st.sidebar.slider("Prediction Range", 1, 60, 15)
    st.sidebar.write("Trains the model based on the selected range ")
    epoch_num = st.sidebar.slider("Number of Epochs", 1, 100, 10)
    st.sidebar.warning("Choosing higher numbers on epochs will result in longer training times. Moreover, training your model for longer periods of time might hinder the generalization of new data")
    company  = st.sidebar.selectbox("Which company you would like to choose?", companies)


    # Store user inputs into a variable
    user_data = {"trained_date": trained_date, "predict_day": predict_day,  "num_epoch": epoch_num, "company": company}

    # Transfrom it into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store user input into a variable
user_input = get_userInput()


