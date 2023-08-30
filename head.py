import streamlit as st
import pickle

# Load your trained model
pickle_a = open("random_forest.pkl", "rb")
regressor = pickle.load(pickle_a)  # our model

def predict_value(headsize, agerange, gender):
    prediction = regressor.predict([[headsize, agerange, gender]])  # predictions using our model
    return prediction

def main():
    st.title("Head Weight Prediction ML App")
    st.markdown(
        """
        <div style='text-align: center;'>
            <h2>Head Weight Prediction ML App</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add an image
    st.image("your_image_path.png", use_column_width=True)

    # Use sliders for user input
    headsize = st.slider("Select Head Size:", 2500, 5000)
    agerange = st.select_slider("Select Age Range:", options=[1, 2])
    gender = st.radio("Select Gender:", ["Male", "Female"])

    # Add a button for prediction
    if st.button("Predict"):
        result = predict_value(headsize, agerange, gender == "Female")
        st.success(f"The estimated head weight is: {result[0]:.2f} grams")

if __name__ == '__main__':
    main()
