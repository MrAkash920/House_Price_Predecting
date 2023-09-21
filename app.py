import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
df = pd.read_csv('Housing.csv')
# Numerical features
numerical_features = ['bedrooms', 'bathrooms', 'area', 'stories','parking']

# Categorical features
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Select the subsets of data
X_numerical = df[numerical_features]
X_categorical = df[categorical_features]
y = df['price']

X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)

X_combined = pd.concat([X_numerical, X_categorical_encoded], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Define a function to predict house prices
def predict_price(bedrooms, bathrooms, area, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus):
    # Create a DataFrame with the user's input
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'area': [area],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    # Apply one-hot encoding to categorical features
    input_data_encoded = pd.get_dummies(input_data, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)

    # Ensure the order and names of columns match the training data
    input_data_encoded = input_data_encoded.reindex(columns=X_combined.columns, fill_value=0)

    # Make predictions using the trained model
    predicted_price = model.predict(input_data_encoded)

    return predicted_price[0]

# Create a Streamlit web app
st.title("Housing Price Prediction")

# User inputs
bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=2)  # Change value=2.5 to value=2
area = st.slider("Area (sq. ft.)", min_value=500, max_value=10000, value=6000)
stories = st.slider("Number of Stories", min_value=1, max_value=5, value=2)
mainroad = st.selectbox("Main Road", ["Yes", "No"])
guestroom = st.selectbox("Guest Room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
parking = st.slider("Number of Parking Spaces", min_value=0, max_value=5, value=1)
prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
furnishingstatus = st.selectbox("Furnishing Status", ["Semi-Furnished", "Unfurnished", "Furnished"])

# Predict button
if st.button("Predict"):
    predicted_price = predict_price(bedrooms, bathrooms, area, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus)
    st.success(f"Predicted Price: Rs{predicted_price:.2f}")
