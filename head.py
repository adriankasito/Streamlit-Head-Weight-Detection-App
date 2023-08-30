import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle
import plotly.express as px

# Add CSS styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2f4f4f;
        padding: 20px;
        font-size: 30px;
        font-weight: bold;
    }
    .header {
        text-align: center;
        color: #4682b4;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #808080;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

data_url = 'headbrain.xlsx'

# Main title
st.markdown("<div class='main-title'>Head Size Detection</div>", unsafe_allow_html=True)

def load_data():
    data = pd.read_excel(data_url)
    data.rename(columns={'age range': 'age_range', 'head size(cm^3)': 'head_size', 'brain weight(grams)': 'brain_weight'}, inplace=True)
    return data

data = load_data()

if st.checkbox('Show Dataset', True):
    st.subheader("Dataset")
    st.dataframe(data, height=300)

# Stylish header
st.markdown("<div class='header'>Dataset Insights</div>", unsafe_allow_html=True)

st.subheader("Dataset Size")
st.write(f"Number of Rows: {data.shape[0]}, Number of Columns: {data.shape[1]}")

# Add color to charts
st.subheader("Breakdown of Head Size and Weight by Gender")
fig_violin = px.violin(data, x='gender', y='brain_weight', points="all", box=True, color='gender', title='Brain Weight Distribution by Gender')
fig_violin.update_traces(marker=dict(size=5, opacity=0.7), line=dict(width=2))
st.plotly_chart(fig_violin)

st.subheader("Relationship between Brain Weight and Head Size")
fig_scatter_age = px.scatter(data, x='head_size', y='brain_weight', color='age_range', title='Brain Weight vs Head Size by Age Range', labels={'head_size': 'Head Size', 'brain_weight': 'Brain Weight'})
fig_scatter_age.update_traces(marker=dict(size=8, opacity=0.7))
st.plotly_chart(fig_scatter_age)

# Add color to model info
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

# Stylish header for app
st.markdown("<div class='header'>Head Weight Prediction App</div>", unsafe_allow_html=True)

with open("random_forest.pkl", "rb") as pkl_file:
    regressor = pickle.load(pkl_file)

def predict_value(head_size, age_range, gender):
    prediction = regressor.predict([[head_size, age_range, gender]])
    return prediction[0]

# Add colorful sliders
st.markdown("<div class='header'>Predict Head Weight</div>", unsafe_allow_html=True)
head_size = st.slider("Head Size (cm³)", 2500, 5000, 3750)
age_range = st.selectbox("Age Range", [1, 2], format_func=lambda x: "Age 1" if x == 1 else "Age 2", index=0)
gender = st.selectbox("Gender", ["Male", "Female"], index=0)

if st.button("Predict"):
    result = predict_value(head_size, age_range, gender)
    st.success(f"The estimated head weight is: {result:.2f} grams")

# Add an image
st.image("head.png", caption="Human Brain Image", use_column_width=True)

# Stylish footer
st.markdown("<div class='footer'>Developed by Your Name &copy; 2023</div>", unsafe_allow_html=True)
