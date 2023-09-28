import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle

data_url = 'headbrain.xlsx'

st.title("Head Size Detection")
st.markdown("This streamlit web application is a dashboard for detecting human head sizes 🗣")

def load_data():
    data = pd.read_excel(data_url)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data.rename(columns={'age range': 'age_range', 'head size(cm^3)': 'head_size', 'head size': 'head_size', 'brain weight(grams)': 'brain_weight'}, inplace=True)
    return data

data = load_data()

if st.checkbox('Show dataset', True):
    st.subheader("Dataset")
    st.write(data)

st.subheader("Size of the dataset")
st.write('data_shape', data.shape)

st.subheader("Breakdown of head size and weight by gender")
fig = px.violin(data, x='gender', y='brain_weight', height=500, width=900, points="all", box=True, color='gender', title='Violin plot with boxes showing breakdown of brain weight within different gender')
newnames = {'1': 'Male', '2': 'Female'}
fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                      legendgroup=newnames[t.name],
                                      hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])))
st.write(fig)

st.subheader("Relationship between brain weight and head size")
fig_1 = px.scatter(data, x='head_size', y='brain_weight', color='age_range', title='Scatter plot showing relationship between brain weight and head size by the age range')
st.write(fig_1.update_traces(showlegend=False))

fig_2 = px.scatter(data, x='head_size', y='brain_weight', color='gender', title='Scatter plot showing relationship between brain weight and head size according to the gender')
st.write(fig_2.update_traces(showlegend=False))

st.subheader("Random Forest Regressor")
X = data[['head_size', 'age_range', 'gender']]
y = data['brain_weight'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

reg = RandomForestRegressor(n_estimators=100, random_state=1)
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
r2_train_score = r2_score(y_train, y_train_pred)

st.write("R-Squared using Random Forest Regressor: ", r2_train_score)

st.write("The model is built, now the model is pickled so that it can be used in the future")
pickle.dump(reg, open('random_forest.pkl', 'wb'))

st.subheader("APP")

pickle_a = open("random_forest.pkl", "rb")
regressor = pickle.load(pickle_a)  # our model

def predict_value(headsize, agerange, gender):
    prediction = regressor.predict([[headsize, agerange, gender]])  # predictions using our model
    return prediction

def main():
    st.title("Head weight prediction APP using ML")
    html_temp = """
        <div>
        <h2>Head Weight Prediction ML App</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    headsize = st.slider("head_size", 2500, 5000)
    agerange = st.slider("age_range", 1, 2)
    gender = st.slider("gender", 1, 2)
    result = ""
    if st.button("Predict"):
        result = predict_value(headsize, agerange, gender)
    st.success("The head weight of that person is: {} grams".format(result))

if __name__ == '__main__':
    main()
