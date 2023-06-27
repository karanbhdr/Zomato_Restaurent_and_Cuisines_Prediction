import streamlit  as st
import pandas as pd
import numpy as np
import pickle
import os 

#load_machine_learning_models
price_predictor_path = r"C:\Users\karan\Downloads\Compressed\Data _Scrapping_Project-20230627T163344Z-001\Data _Scrapping_Project\Zomato_Project\Machine_learning_model\price_predictor"
cusines_predictor_path = r"C:\Users\karan\Downloads\Compressed\Data _Scrapping_Project-20230627T163344Z-001\Data _Scrapping_Project\Zomato_Project\Machine_learning_model\Cuisine_predictor"

load_model = pickle.load(open(price_predictor_path, 'rb'))
cusine_model = pickle.load(open(cusines_predictor_path, 'rb'))


# load_DATASET_mapping
df = pd.read_excel(r"C:\Users\karan\Downloads\Compressed\Data _Scrapping_Project-20230627T163344Z-001\Data _Scrapping_Project\Zomato_Project\Dataset\Combined_data.xlsx")
loc_df = pd.read_excel(r"C:\Users\karan\Downloads\Compressed\Data _Scrapping_Project-20230627T163344Z-001\Data _Scrapping_Project\Zomato_Project\Dataset\frq_dis_loc_map.xlsx")
cu_path = pd.read_excel(r"C:\Users\karan\Downloads\Compressed\Data _Scrapping_Project-20230627T163344Z-001\Data _Scrapping_Project\Zomato_Project\Dataset\frq_dis_cus_map.xlsx")

# define_ui amd layout 
st.title("Restaurent Selector")
column_left, column_right = st.columns(2)

#Left columns
with column_left:
    filter_location = st.selectbox("Prefferd Location:", options=df['Location'].unique())

    #BEST-CUSINES
    best_cuisine = df[df['Location'] == filter_location].groupby('Cuisines')['Delivery_review_number'].sum().idxmax()
    st.write(f"For the selected area, the most popular cuisine is: **{best_cuisine}**.")

    # Calculate average price for 1 person for selected location
    filter_location_result = df[df['Location'] == filter_location]
    avg_price = filter_location_result['Price_For_One'].mean()
    st.write(f"And the average price for one person is: **₹{avg_price:.2f}**.")

    # Find best restaurant and list of cuisines they serve for selected location
    highest_rating = filter_location_result['Delivery_review_number'].max()
    best_restaurant = df[(df['Location'] == filter_location) & (df['Delivery_review_number'] == highest_rating)]
    cuisine_list = best_restaurant['Cuisines'].tolist()
    restaurant_name = best_restaurant['Restaurant_Name'].unique().tolist()
    st.write(f"For the selected location, the most popular restaurant is: **{restaurant_name[0]}** and cuisines that they serve are: **{cuisine_list}**.")



# Right column: cuisine-based recommendations
with column_right:
    # Dropdown menu for preferred cuisine
    filter_cuisine = st.selectbox('Preferred Cuisine:', options=df['Cuisines'].unique())

    # Find best restaurant and location for selected cuisine
    highest_rating_cuisine = df[df['Cuisines'] == filter_cuisine]['Delivery_review_number'].max()
    best_restaurant_cuisine = df[(df['Cuisines'] == filter_cuisine) & (df['Delivery_review_number'] == highest_rating_cuisine)]
    st.write(f"For the selected cuisine, the most popular restaurant is: **{best_restaurant_cuisine['Restaurant_Name'].iloc[0]}** in **{best_restaurant_cuisine['Location'].iloc[0]}**.")


filter_price = st.slider('Preferred price for one:', min_value=50, max_value=500, value=50, step=50)
#mapping location and cuisines
user_input_location = filter_location
matching_row = loc_df.loc[loc_df['Location'] == user_input_location]
index = matching_row.index[0]
value_location = loc_df.iloc[index, 1]

user_input_cuisine = filter_cuisine
matching_row = cu_path.loc[cu_path['Cuisines'] == user_input_cuisine]
index = matching_row.index[0]
value_cuisine = cu_path.iloc[index, 1]

def price_prediction(input_data):
    input_data_as_numpy = np.asarray(input_data)
    input_array_reshape = input_data_as_numpy.reshape(1,-1)
    predict_price = load_model.predict(input_array_reshape)
    return predict_price

columnleft,columnright = st.columns(2)
price = price_prediction([value_location,value_cuisine])
with columnleft:
    st.header("Recommended Price")

    st.write(f"Recommended Price according to your preference is: {price}")

def cuisine_prediction(input_data):
    input_data_as_numpy = np.asarray(input_data)
    input_array_reshape = input_data_as_numpy.reshape(1,-1)
    predict_cuisine = cusine_model.predict(input_array_reshape)
    return predict_cuisine

cuisine = cuisine_prediction([filter_price,value_location])
with columnright:
    st.header("Recommended Cuisine")

    st.write(f"Recommended Cuisine according to your preference is: {cuisine}")

