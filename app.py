
import streamlit as st
import datetime

st.write("This program will never be able to predict the future values exactly. Although, It might give an intuition towards the current and potential future trend of a specific stock.")

# Date
st.title("Date range")

min_date = datetime.datetime(2015, 1, 1)
max_date = datetime.date(2020, 1, 1)

date = st.sidebar.date_input('start date', datetime.date(2011,1,1))


