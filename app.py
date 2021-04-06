
import streamlit as st
import datetime
import pandas as pd
import datapackage
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

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

# Data Input
company  = user_input["company"]
st.write(company[0])
df = web.DataReader(company[0], data_source="yahoo", start=start, end=end)
df1 = df.reset_index()["Close"]


# visualize
st.line_chart(df.Close)
st.line_chart(df.Volume)

#PREDICTION

# Min - Max scaler

scaler = MinMaxScaler(feature_range=(0, 1))

df1 = scaler.fit_transform((np.array(df1).reshape(-1, 1)))
# Split

training_size = int(len(df1) * 0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0: training_size, :], df1[training_size: len(df1), :1]


# Convert into dataset


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []

    for i in range(len(dataset) - time_step - 1):
        a = dataset[i: (i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


trained_date = int(user_input["trained_date"])
X_train, y_train = create_dataset(train_data, trained_date)
X_test, y_test = create_dataset(test_data, trained_date)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=int(user_input["num_epoch"]),batch_size=64,verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


last_day = 100
x_input_range = len(test_data) - last_day
x_input = test_data[x_input_range:].reshape(1,-1)

temp_input = list(x_input)
temp_input = temp_input[0].tolist()


lst_output = []
n_steps = last_day
i = 0
predict_day = int(user_input["predict_day"])

while(i < predict_day):

    if(len(temp_input) > last_day):
        x_input = np.array((temp_input[1:]))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1
st.table(scaler.inverse_transform(lst_output))



