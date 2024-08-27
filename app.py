import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model from Google Drive
model_path = '/content/dl_assignment_model2.h5'  # Update with your model path
model = load_model(model_path, compile=False)  # Avoid recompiling

# Load the fitted scaler
scaler = joblib.load('/content/scaler.pkl')  # Update with your scaler path

# Create a function to preprocess user input
def preprocess_input(data):
    # Use the fitted scaler to transform the input data
    scaled_data = scaler.transform(data)
    return scaled_data

# Streamlit application
def main():
    st.title("Student Performance Prediction")

    # Sidebar for user input
    st.sidebar.header("User Input Features")

    def user_input_features():
        # Collect user inputs for the selected features
        course_id = st.sidebar.number_input('Course ID', min_value=1, max_value=100, value=10)
        gpa_last_sem = st.sidebar.slider('GPA in Last Semester (/4.00)', min_value=0.0, max_value=4.0, value=3.0)
        mothers_education = st.sidebar.selectbox('Mother\'s Education', [0, 1, 2, 3, 4])
        num_siblings = st.sidebar.number_input('Number of Siblings', min_value=0, max_value=10, value=2)
        fathers_education = st.sidebar.selectbox('Father\'s Education', [0, 1, 2, 3, 4])
        expected_gpa_graduation = st.sidebar.slider('Expected GPA at Graduation (/4.00)', min_value=0.0, max_value=4.0, value=3.5)
        fathers_occupation = st.sidebar.selectbox('Father\'s Occupation', [0, 1])
        study_hours = st.sidebar.slider('Weekly Study Hours', min_value=0, max_value=100, value=10)
        taking_notes = st.sidebar.selectbox('Taking Notes in Classes', [0, 1])
        flip_classroom = st.sidebar.selectbox('Flip-Classroom', [0, 1])

        data = {
            'COURSE ID': [course_id],
            'Cumulative grade point average in the last semester (/4.00)': [gpa_last_sem],
            'Mothers education': [mothers_education],
            'No. of siblings': [num_siblings],
            'Father education': [fathers_education],
            'Expected Cumulative grade point average in the graduation (/4.00)': [expected_gpa_graduation],
            'Father occupation': [fathers_occupation],
            'Weekly study hours': [study_hours],
            'Taking notes in classes': [taking_notes],
            ' Flip-classroom': [flip_classroom]
        }
        return pd.DataFrame(data)

    input_features = user_input_features()

    st.subheader("User Input:")
    st.write(input_features)

    # Preprocess the input data
    preprocessed_data = preprocess_input(input_features)

    # Make prediction
    prediction = model.predict(preprocessed_data)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map predicted class to label
    class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7']  # Update with your class labels
    st.subheader('Prediction')
    st.write(f'Predicted class: {class_labels[predicted_class]}')

if __name__ == "__main__":
    main()
