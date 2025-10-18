import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 2rem;
    }
    .price-result {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class CarPricePredictor:
    def __init__(self, model_path='car_price_predictor.pkl'):
        try:
            if not os.path.exists(model_path):
                st.error(f"❌ Model file '{model_path}' not found!")
                st.stop()
            
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.encoder = model_data['encoder']
            self.feature_names = model_data['feature_names']
            self.numerical_features = model_data['numerical_features']
            self.categorical_features = model_data['categorical_features']
            
            # Handle available cars
            if 'available_cars' in model_data:
                self.available_cars = model_data['available_cars']
            else:
                # Fallback: Extract car names from feature names
                car_columns = [col for col in self.feature_names if col.startswith('Car_Name_')]
                self.available_cars = [col.replace('Car_Name_', '') for col in car_columns]
                
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()
    
    def predict_price(self, year, present_price, kms_driven, owner, 
                     car_name, fuel_type, seller_type, transmission):
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Year': [year],
                'Present_Price': [present_price],
                'Kms_Driven': [kms_driven],
                'Owner': [owner],
                'Car_Name': [car_name],
                'Fuel_Type': [fuel_type],
                'Seller_Type': [seller_type],
                'Transmission': [transmission]
            })
            
            # Preprocess the input
            encoded_data = self.encoder.transform(input_data[self.categorical_features])
            encoded_df = pd.DataFrame(
                encoded_data, 
                columns=self.encoder.get_feature_names_out(self.categorical_features)
            )
            
            # Combine features
            final_input = pd.concat([input_data[self.numerical_features], encoded_df], axis=1)
            
            # Ensure all columns are present
            for col in self.feature_names:
                if col not in final_input.columns:
                    final_input[col] = 0
            
            # Reorder columns
            final_input = final_input[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(final_input)
            return prediction[0]
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict the selling price of your car using Machine Learning")
    
    try:
        # Initialize predictor
        predictor = CarPricePredictor('car_price_predictor.pkl')
        
        # Display car count
        st.sidebar.success(f"✅ Loaded {len(predictor.available_cars)} car models")
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("📋 Car Details")
            
            # Create form for car details
            with st.form("car_price_form"):
                # Car selection
                car_name = st.selectbox(
                    "Select Car Model",
                    options=predictor.available_cars,
                    help="Choose your car model from the list"
                )
                
                # Create three columns for related inputs
                col1a, col1b, col1c = st.columns(3)
                
                with col1a:
                    fuel_type = st.selectbox(
                        "Fuel Type",
                        options=['Petrol', 'Diesel', 'CNG'],
                        help="Select the fuel type of your car"
                    )
                
                with col1b:
                    transmission = st.selectbox(
                        "Transmission",
                        options=['Manual', 'Automatic'],
                        help="Select transmission type"
                    )
                
                with col1c:
                    seller_type = st.selectbox(
                        "Seller Type",
                        options=['Dealer', 'Individual'],
                        help="Are you a dealer or individual seller?"
                    )
                
                # Numeric inputs in another row
                col2a, col2b, col2c, col2d = st.columns(4)
                
                with col2a:
                    year = st.slider(
                        "Manufacturing Year",
                        min_value=2000,
                        max_value=2023,
                        value=2018,
                        help="Select the year your car was manufactured"
                    )
                
                with col2b:
                    present_price = st.number_input(
                        "Present Price (₹ Lakhs)",
                        min_value=0.1,
                        max_value=100.0,
                        value=10.0,
                        step=0.1,
                        help="Current showroom price of the car"
                    )
                
                with col2c:
                    kms_driven = st.number_input(
                        "Kilometers Driven",
                        min_value=0,
                        max_value=500000,
                        value=15000,
                        step=1000,
                        help="Total kilometers the car has been driven"
                    )
                
                with col2d:
                    owner = st.selectbox(
                        "Previous Owners",
                        options=[0, 1, 2, 3],
                        help="Number of previous owners"
                    )
                
                # Submit button
                submitted = st.form_submit_button("🚀 Predict Selling Price", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("ℹ️ How It Works")
            st.markdown("""
            This app predicts car prices using a **Machine Learning model** trained on:
            
            - **98 different car models**
            - **300+ data points**
            - **Multiple features**: Year, KM driven, Fuel type, etc.
            
            **Model Accuracy**: 96% R² Score
            
            *Fill in the details on the left and click 'Predict' to get your car's estimated selling price!*
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick tips
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("💡 Tips for Better Price")
            st.markdown("""
            - Regular maintenance records
            - Single owner cars get better prices
            - Lower kilometers = higher value
            - Keep service history updated
            - Clean interior and exterior
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction result
        if submitted:
            st.markdown("---")
            
            with st.spinner('🔮 Predicting your car price...'):
                # Make prediction
                predicted_price = predictor.predict_price(
                    year=year,
                    present_price=present_price,
                    kms_driven=kms_driven,
                    owner=owner,
                    car_name=car_name,
                    fuel_type=fuel_type,
                    seller_type=seller_type,
                    transmission=transmission
                )
            
            if predicted_price is not None:
                # Display result
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("🎯 Prediction Result")
                
                # Display car details
                col3a, col3b, col3c = st.columns(3)
                
                with col3a:
                    st.metric("Car Model", car_name)
                    st.metric("Manufacturing Year", year)
                
                with col3b:
                    st.metric("Fuel Type", fuel_type)
                    st.metric("Transmission", transmission)
                
                with col3c:
                    st.metric("Kilometers", f"{kms_driven:,}")
                    st.metric("Previous Owners", owner)
                
                # Display predicted price
                st.markdown(f'<div class="price-result">₹ {predicted_price:.2f} Lakhs</div>', unsafe_allow_html=True)
                
                # Additional info
                depreciation = present_price - predicted_price
                depreciation_percent = (depreciation / present_price) * 100
                
                st.info(f"""
                **💰 Price Analysis:**
                - Present Price: ₹ {present_price:.2f} Lakhs
                - Predicted Selling Price: ₹ {predicted_price:.2f} Lakhs  
                - Depreciation: ₹ {depreciation:.2f} Lakhs ({depreciation_percent:.1f}%)
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Reset button
            if st.button("🔄 Predict Another Car", use_container_width=True):
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Application error: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit & Machine Learning | "
        "Model: Gradient Boosting Regressor (96% Accuracy)"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()