
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


