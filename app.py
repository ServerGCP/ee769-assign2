from warnings import filterwarnings as fw
import streamlit as st
import pickle
import random
fw('ignore')

def main():
    # Title of the web app
    st.title('White Wine Quality Predictor')
    
    # Load data
    white_wine_data = pickle.load(open('white_wine_data.pkl', 'rb'))
    df = white_wine_data['df']
    scaler = white_wine_data['scaler']
    model = white_wine_data['model']
    
    # Session state to store slider values
    slider_dict = st.session_state.slider if 'slider' in st.session_state else {}
    
    # Description
    st.write('Enter the characteristics of the wine to predict its quality.')
    
    # Function to create sliders
    def get_sliders(rand=False):
        random_row = df.sample()
        for column in df.columns[:-1]:
            min_val = df[column].min()
            max_val = df[column].max()
            step = round((max_val - min_val) / 100, 3)
            try:
                if rand or 'slider' not in st.session_state:
                    # Create slider with random value
                    slider_dict[column] = st.slider(column, min_value=min_val, max_value=max_val, step=step, value=random_row[column].values[0])
                else:
                    # Use previous value if available
                    slider_dict[column] = st.slider(column, min_value=min_val, max_value=max_val, step=step, value=slider_dict[column])
            except: pass
    
    # Generate sliders if Random button is clicked or for initial load
    if st.button('Random') or 'init' not in st.session_state:
        st.session_state.init = False
        get_sliders(True)
        st.session_state.slider = slider_dict
    
    # Display sliders
    get_sliders(False)
    
    # Prediction button
    if st.button('Predict'):
        # Transform features and make prediction
        features = [list(slider_dict.values())]
        features = scaler.transform(features)
        quality = model.predict(features)[0]
        # Display predicted quality with success message
        st.success(f'The predicted wine quality is {quality}')
