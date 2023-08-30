import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle

data_url = 'headbrain.xlsx'

st.title("Head Size Detection")
st.markdown("<h2 style='text-align: center;'>Human Head Size Detection Dashboard</h2>", unsafe_allow_html=True)

def load_data():
    data = pd.read_excel(data_url)
    data.rename(columns={'age range': 'age_range', 'head size(cm^3)': 'head_size', 'brain weight(grams)': 'brain_weight'}, inplace=True)
    return data

data = load_data()

if st.checkbox('Show Dataset', True):
    st.subheader("Dataset")
    st.write(data)

st.subheader("Dataset Size")
st.write(f"Number of Rows: {data.shape[0]}, Number of Columns: {data.shape[1]}")

st.subheader("Breakdown of Head Size and Weight by Gender")
st.plotly_chart(px.violin(data, x='gender', y='brain_weight', points="all", box=True, color='gender', title='Brain Weight Distribution by Gender'))

st.subheader("Relationship between Brain Weight and Head Size")
st.plotly_chart(px.scatter(data, x='head_size', y='brain_weight', color='age_range', title='Brain Weight vs Head Size by Age Range', labels={'head_size': 'Head Size', 'brain_weight': 'Brain Weight'}))

st.subheader("Random Forest Regressor")
X = data[['head_size', 'age_range', 'gender']]
y = data['brain_weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

reg = RandomForestRegressor(n_estimators=100, random_state=1)
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
r2_train_score = r2_score(y_train, y_train_pred)

st.write(f"R-Squared using Random Forest Regressor: {r2_train_score:.2f}")

st.write("The model is built and saved for future use")
with open('random_forest.pkl', 'wb') as pkl_file:
    pickle.dump(reg, pkl_file)

st.subheader("Head Weight Prediction App")

with open("random_forest.pkl", "rb") as pkl_file:
    regressor = pickle.load(pkl_file)

def predict_value(head_size, age_range, gender):
    prediction = regressor.predict([[head_size, age_range, gender]])
    return prediction

st.markdown("<h3 style='text-align: center;'>Predict Head Weight Using Machine Learning</h3>", unsafe_allow_html=True)

head_size = st.slider("Head Size (cm^3)", 2500, 5000)
age_range = st.slider("Age Range", 1, 2)
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):
    result = predict_value(head_size, age_range, gender)
    st.success(f"The estimated head weight is: {result[0]:.2f} grams")

# Add an image
st.image("your_image_path.png", caption="Human Brain Image", use_column_width=True)
