import streamlit as st

from src.predict import predict_house_price


st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 House Price Prediction Dashboard")

st.write(
    "Enter California housing features below to generate a predicted median house value."
)

st.info(
    "Prediction values use the California Housing dataset scale, where 1.0 represents approximately $100,000."
)

st.sidebar.header("Model Input Features")

med_inc = st.sidebar.number_input(
    "Median Income",
    min_value=0.0,
    value=8.3252,
    step=0.1,
)

house_age = st.sidebar.number_input(
    "House Age",
    min_value=0.0,
    value=41.0,
    step=1.0,
)

ave_rooms = st.sidebar.number_input(
    "Average Rooms",
    min_value=0.0,
    value=6.984127,
    step=0.1,
)

ave_bedrms = st.sidebar.number_input(
    "Average Bedrooms",
    min_value=0.0,
    value=1.023810,
    step=0.1,
)

population = st.sidebar.number_input(
    "Population",
    min_value=0.0,
    value=322.0,
    step=10.0,
)

ave_occup = st.sidebar.number_input(
    "Average Occupancy",
    min_value=0.0,
    value=2.555556,
    step=0.1,
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=37.88,
    step=0.01,
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=-122.23,
    step=0.01,
)

input_data = {
    "MedInc": med_inc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrms,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude,
}

if st.button("Predict House Value"):
    result = predict_house_price(input_data)
    predicted_value = result["predicted_house_value"]
    estimated_dollars = predicted_value * 100000

    st.success(f"Predicted House Value: ${estimated_dollars:,.2f}")
    st.caption(f"Raw model output: {predicted_value}")

st.divider()

st.subheader("Model Notes")
st.write(
    """
    This dashboard uses the trained Random Forest model saved in the project `models/` directory.
    It is designed as a lightweight interface for demonstrating model inference in a production-style ML workflow.
    """
)