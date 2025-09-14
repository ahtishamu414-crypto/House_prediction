import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load model & encoder
# ------------------------------
model = joblib.load("house_price_model.pkl")
le = joblib.load("location_encoder.pkl")

# ------------------------------
# Helper function
# ------------------------------
def area_to_marla(area):
    """Convert area input to numeric Marla."""
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
# Page Config
# ------------------------------
st.set_page_config(page_title="🏡 Lahore House Price Predictor", page_icon="🏠", layout="wide")

# ------------------------------
# Title Section
# ------------------------------
st.title("🏡 Lahore House Price Prediction")
st.markdown(
    "<p style='color:gray; font-size:18px;'>"
    "Enter house details to estimate the price in Crores."
    "</p>", unsafe_allow_html=True
)

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("📌 Input House Details")

location = st.sidebar.selectbox("Select Location", le.classes_)
area = st.sidebar.text_input("Area (e.g., 5 Marla, 1 Kanal)", "1 Kanal")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 6)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 7)
built_year = st.sidebar.number_input("Built Year", min_value=1960, max_value=2025, value=2024)
kitchens = st.sidebar.slider("Kitchens", 0, 5, 2)
store_rooms = st.sidebar.slider("Store Rooms", 0, 5, 1)
servant_quarters = st.sidebar.slider("Servant Quarters", 0, 5, 1)

# Checkboxes grouped
st.sidebar.subheader("🏠 Features")
furnished = st.sidebar.checkbox("Furnished", True)
gym = st.sidebar.checkbox("Gym", False)
study_room = st.sidebar.checkbox("Study Room", False)
drawing_room = st.sidebar.checkbox("Drawing Room", True)
dining_room = st.sidebar.checkbox("Dining Room", True)
lawn_garden = st.sidebar.checkbox("Lawn/Garden", True)
swimming_pool = st.sidebar.checkbox("Swimming Pool", False)
electricity_backup = st.sidebar.checkbox("Electricity Backup", True)
lounge = st.sidebar.checkbox("Lounge/Sitting Room", True)

# ------------------------------
# Prediction
# ------------------------------
if st.sidebar.button("🔮 Predict Price"):
    try:
        # Encode location
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

        # Predict
        price_pred = model.predict(input_df)[0]

        # Display result
        price_crore = price_pred / 1e7
        st.markdown(
            f"""
            <div style='background-color:#e6f7ff; padding:20px; border-radius:12px;'>
                <h2 style='color:#007acc;'>💰 Estimated Price: {price_crore:.2f} Crore</h2>
            </div>
            """, unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
