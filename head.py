import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

data_url= ('headbrain.xlsx')

st.title("Head Size Detection")
st.markdown("This streamlit web application is a dashboard for detecting human head sizes 🗣")

#@st.cache(persist=True)
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
fig = px.violin(data, x='gender', y='brain_weight', height = 500, width= 900, points= "all", box = True, color='gender',title='Violin plot with boxes showing breakdown of brain weight within different gender')
newnames = {'1':'Male', '2': 'Female'}
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
st.write(fig)

st.subheader("Relationship between brain weight and head size")
fig_1 = px.scatter(data, x='head_size', y= 'brain_weight', color='age_range',title='Scatter plot showing relationship between brain weight and head size by the age range')
st.write(fig_1)

fig_2 = px.scatter(data, x='head_size', y= 'brain_weight', color='gender',title='Scatter plot showing relationship between brain weight and head size according to the gender')
st.write(fig_2)

st.subheader("Linear Model")
X1 = data['head_size'].values
y1 = data['brain_weight'].values

#mean of X and y
mean_x = np.mean(X1)
mean_y = np.mean(y1)

#total no of val
m = len(X1)

#finding b0 and b1
numer = 0
denom = 0
for i in range(m):
    numer += (X1[i] - mean_x) * (y1[i]- mean_y)
    denom += (X1[i] - mean_x) ** 2
b1 = numer/denom
b0 = mean_y - (b1 * mean_x)

st.write("Intercept of the regression line", b0)
st.write("Slope of the regression line", b1)

import matplotlib.pyplot as plt
max_x = np.max(X1) + 100
min_x = np.min(X1) - 100

x = np.linspace(min_x, max_x, 1000)
y_ = b0 +b1*x


fig1 = px.line(x=x, y=y_)
fig1.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))


fig2 = px.scatter(x=X1, y=y1, symbol= data.age_range)
newname = {'1':'Lower Ages', '2': 'Upper Ages'}
fig2.for_each_trace(lambda t: t.update(name = newname[t.name],
                                      legendgroup = newname[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newname[t.name])))


fig3 = go.Figure(data=fig1.data + fig2.data)
st.write(fig3)

# linear model using least quare method
# to check how good our model is

ss_t = 0
ss_r = 0
for i in range (m):
    y_pred = b0 + b1 *X1[i]
    ss_t += (y1[i] - mean_y) ** 2 #total sum of square
    ss_r += (y1[i] - y_pred) ** 2 #total sum of square of residuals
r2 = 1- (ss_r/ ss_t)
st.write("R-Squared using Linear Squares Method: ",r2)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = data[['head_size', 'age_range', 'gender']]
y = data['brain_weight'].values
# cannot use rank 1 matric in scikit learn
#X = X.reshape(m, 1)

st.write("Train test split of sklearn library is used to divide the data into train and test. The train set is used to train the algorithm/model. The test set is used to evaluate how our model is predicting.")

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.30,random_state=1)
#creating model
reg = LinearRegression()

# fitting training data
reg = reg.fit(X_train, y_train)

# Y prediction
#Y_pred = reg.predict(X_test)
#df1 = pd.DataFrame(X_test)
#df2 = pd.DataFrame(Y_pred, columns=['predicted head weight'])
#df3 = pd.concat([df1, df2], axis=1, ignore_index=False)
#st.write("Predictions :",df3)

st.write("The model is built, now the model is pickled so that it can be used in future")
import pickle
pickle.dump(reg,open('linear.pkl','wb'))


# Calculating R2 Score
r2_score = reg.score(X_train, y_train)

st.write("R-Squared using sklearn: ", r2_score)

st.subheader("APP")

pickle_a=open("linear.pkl","rb")
regressor=pickle.load(pickle_a) # our model

def predict_value(headsize, agerange, Gender):
    prediction=regressor.predict([[headsize, agerange, Gender]]) #predictions using our model
    return prediction


def main():
    st.title("Head weight prediction APP using ML") #simple title for the app
    html_temp="""
        <div>
        <h2>Head Weight Prediction ML App</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True) #a simple html
    headsize=st.slider("head_size", 2500, 5000) #giving inputs as used in building the model
    agerange = st.slider("age_range", 1,2)
    Gender = st.slider("gender", 1,2)
    result=""
    if st.button("Predict"):
        result=predict_value(headsize, agerange, Gender) #result will be displayed if button is pressed
    st.success("The head weight of that person is :{}".format(result))

if __name__=='__main__':
    main()
