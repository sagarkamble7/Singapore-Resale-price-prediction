from PIL import Image
import pickle
from streamlit_option_menu import option_menu
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def town_mapping(town_map):
    mapping = {
        'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4, 'BUKIT PANJANG': 5,
        'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8, 'CLEMENTI': 9, 'GEYLANG': 10,
        'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13, 'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15,
        'PASIR RIS': 16, 'PUNGGOL': 17, 'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'SENGKANG': 20,
        'SERANGOON': 21, 'TAMPINES': 22, 'TOA PAYOH': 23, 'WOODLANDS': 24, 'YISHUN': 25
    }
    # Return -1 or any default value if not found
    return mapping.get(town_map, -1)


def flat_type_mapping(flt_type):
    mapping = {
        '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4, '2 ROOM': 1, 'EXECUTIVE': 5,
        '1 ROOM': 0, 'MULTI-GENERATION': 6
    }
    # Return -1 or any default value if not found
    return mapping.get(flt_type, -1)


def flat_model_mapping(fl_m):
    mapping = {
        'Improved': 5, 'New Generation': 12, 'Model A': 8, 'Standard': 17, 'Simplified': 16,
        'Premium Apartment': 13, 'Maisonette': 7, 'Apartment': 3, 'Model A2': 10, 'Type S1': 19,
        'Type S2': 20, 'Adjoined flat': 2, 'Terrace': 18, 'DBSS': 4, 'Model A-Maisonette': 9,
        'Premium Maisonette': 15, 'Multi Generation': 11, 'Premium Apartment Loft': 14,
        'Improved-Maisonette': 6, '2-room': 0, '3Gen': 1
    }
    return mapping.get(fl_m, -1)  # Return -1 or any default value if not found


def predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, re_les_year,
                  re_les_month, les_coms_dt):

    # Convert inputs to appropriate types
    year_1 = int(year)
    town_2 = town_mapping(town)
    flt_ty_2 = flat_type_mapping(flat_type)
    flr_ar_sqm_1 = float(flr_area_sqm)
    flt_model_2 = flat_model_mapping(flat_model)

    # Handling log transformation, adding 1 to avoid log(0) issues
    str_str = np.log(float(stry_start) + 1)
    str_end = np.log(float(stry_end) + 1)

    rem_les_year = int(re_les_year)
    rem_les_month = int(re_les_month)
    lese_coms_dt = int(les_coms_dt)

    with open(r"D:\Singapore resale prioce\Resale_Flat_Prices_Model_1.pkl", "rb") as f:
        regg_model = pickle.load(f)

    user_data = np.array([[year_1, town_2, flt_ty_2, flr_ar_sqm_1,
                           flt_model_2, str_str, str_end, rem_les_year, rem_les_month,
                           lese_coms_dt]])

    # Ensure data is finite
    if not np.all(np.isfinite(user_data)):
        st.error("Input data contains invalid or infinite values.")
        return None

    y_pred_1 = regg_model.predict(user_data)
    price = np.exp(y_pred_1[0])

    return round(price)


st.set_page_config(layout="wide")

st.title("SINGAPORE RESALE FLAT PRICES PREDICTING")
st.write("")

with st.sidebar:
    select = option_menu("MAIN MENU", ["Home", "Price Prediction"])

if select == "Home":

    st.header("HDB Flats:")

    st.write('''The majority of Singaporeans live in public housing provided by the HDB.
    HDB flats can be purchased either directly from the HDB as a new unit or through the resale market from existing owners.''')

    st.header("Resale Process:")

    st.write('''In the resale market, buyers purchase flats from existing flat owners, and the transactions are facilitated through the HDB resale process.
    The process involves a series of steps, including valuation, negotiations, and the submission of necessary documents.''')

    st.header("Valuation:")

    st.write('''The HDB conducts a valuation of the flat to determine its market value. This is important for both buyers and sellers in negotiating a fair price.''')

    st.header("Eligibility Criteria:")

    st.write("Buyers and sellers in the resale market must meet certain eligibility criteria, including citizenship requirements and income ceilings.")

    st.header("Resale Levy:")

    st.write("For buyers who have previously purchased a subsidized flat from the HDB, there might be a resale levy imposed when they purchase another flat from the HDB resale market.")

    st.header("Grant Schemes:")

    st.write("There are various housing grant schemes available to eligible buyers, such as the CPF Housing Grant, which provides financial assistance for the purchase of resale flats.")

    st.header("HDB Loan and Bank Loan:")

    st.write("Buyers can choose to finance their flat purchase through an HDB loan or a bank loan. HDB loans are provided by the HDB, while bank loans are obtained from commercial banks.")

    st.header("Market Trends:")

    st.write("The resale market is influenced by various factors such as economic conditions, interest rates, and government policies. Property prices in Singapore can fluctuate based on these factors.")

    st.header("Online Platforms:")

    st.write("There are online platforms and portals where sellers can list their resale flats, and buyers can browse available options.")

elif select == "Price Prediction":

    col1, col2 = st.columns(2)
    with col1:

        year = st.selectbox("Select the Year", ["2015", "2016", "2017", "2018", "2019", "2020", "2021",
                                                "2022", "2023", "2024"])

        town = st.selectbox("Select the Town", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                                'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                                'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                                'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                                'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                                'TOA PAYOH', 'WOODLANDS', 'YISHUN'])

        flat_type = st.selectbox("Select the Flat Type", ['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM',
                                                          'MULTI-GENERATION'])

        flr_area_sqm = st.number_input(
            "Enter the Value of Floor Area sqm (Min: 31 / Max: 280")

        flat_model = st.selectbox("Select the Flat Model", ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                                                            'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                                                            'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                                                            'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                                                            'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'])

    with col2:

        stry_start = st.number_input(
            "Enter the Value of Storey Range Start (Min: 1 / Max: 50)")

        stry_end = st.number_input(
            "Enter the Value of Storey Range End (Min: 1 / Max: 50)")

        re_les_year = st.number_input(
            "Enter the Value of Remaining Lease Year (Min: 0 / Max: 99)")

        re_les_month = st.number_input(
            "Enter the Value of Remaining Lease Month (Min: 0 / Max: 11)")

        les_coms_dt = st.number_input(
            "Enter the Value of Lease Commence Date (Min: 1966 / Max: 2023)")

    if st.button("Predict Price"):

        predicted_price = predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end,
                                        re_les_year, re_les_month, les_coms_dt)

        if predicted_price is not None:
            pass
            st.success(f"The Predicted Price of Resale Flat: ${
                predicted_price}")
        else:
            st.error("Prediction failed due to invalid input data.")
