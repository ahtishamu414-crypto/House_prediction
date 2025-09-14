import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load trained model, encoder, and scaler
# ------------------------------
model = joblib.load("house_price_model.pkl")
le = joblib.load("location_encoder.pkl")   # LabelEncoder saved during training
scaler = joblib.load("scaler.pkl")        # Scaler saved during training

# ------------------------------
# Helper function
# ------------------------------
def area_to_marla(area):
    """Convert area input (Marla/Kanal) to numeric Marla."""
    area = area.strip()
    if 'Kanal' in area:
        return float(area.replace('Kanal', '').strip()) * 20
    elif 'Marla' in area:
        return float(area.replace('Marla', '').strip())
    else:
        try:
            return float(area)
        except:
            return 0

# ------------------------------
# App title
# ------------------------------
st.title("🏠 Lahore House Price Prediction")

# ------------------------------
# User Inputs
# ------------------------------
# Dropdown for Location (only trained locations)
location = st.selectbox("Select Location", options=le.classes_)

area = st.text_input("Area (e.g., 5 Marla, 1 Kanal)", "1 Kanal")
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=6)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=7)
built_year = st.number_input("Built Year", min_value=1960, max_value=2025, value=2024)
kitchens = st.number_input("Kitchens", min_value=0, max_value=5, value=3)
store_rooms = st.number_input("Store Rooms", min_value=0, max_value=5, value=2)
servant_quarters = st.number_input("Servant Quarters", min_value=0, max_value=5, value=2)

furnished = st.checkbox("Furnished", True)
gym = st.checkbox("Gym", True)
study_room = st.checkbox("Study Room", True)
drawing_room = st.checkbox("Drawing Room", True)
dining_room = st.checkbox("Dining Room", True)
lawn_garden = st.checkbox("Lawn/Garden", True)
swimming_pool = st.checkbox("Swimming Pool", True)
electricity_backup = st.checkbox("Electricity Backup", True)
lounge = st.checkbox("Lounge/Sitting Room", True)

# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict Price"):
    try:
        # Encode Location
        location_encoded = le.transform([location])[0]

        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            "Location": location_encoded,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Built Year": built_year,
            "Kitchens": kitchens,
            "Store Rooms": store_rooms,
            "Servant Quarters": servant_quarters,
            "Furnished": int(furnished),
            "Gym": int(gym),
            "Study Room": int(study_room),
            "Drawing Room": int(drawing_room),
            "Dining Room": int(dining_room),
            "Lawn/Garden": int(lawn_garden),
            "Swimming Pool": int(swimming_pool),
            "Electricity Backup": int(electricity_backup),
            "Lounge/Sitting Room": int(lounge),
            "Area_cleaned": area_to_marla(area)
        }])

        # Apply scaler to numeric columns
        numeric_cols = ['Location','Bedrooms','Bathrooms','Built Year','Kitchens',
                        'Store Rooms','Servant Quarters','Area_cleaned']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        price_pred = model.predict(input_df)[0]

        # Convert to Crore
        price_crore = price_pred / 1e7
        st.success(f"💰 Predicted House Price: {price_crore:.2f} Crore")

    except Exception as e:
        st.error(f"❌ Error: {e}")
