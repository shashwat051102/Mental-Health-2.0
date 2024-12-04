import pandas as pd
import numpy as np
import pickle
import streamlit as st

from PIL import Image


pickle_in = open('final_model.pkl', 'rb')
classifier = pickle.load(pickle_in)




def prediction(Have_you_ever_had_suicidal_thoughts_,Work_Study_Hours,Financial_Stress, Pressure):
    
    prediction = classifier.predict([[Have_you_ever_had_suicidal_thoughts_,Work_Study_Hours,Financial_Stress,Pressure]])
    print(prediction)
    return prediction



def main():
    # st.title("Mental Health Prediction")
    
    html_temp = """
    <div padding:8px"> 
    <h1 style ="color:white;text-align:center;">Mental Health Prediction App</h1> 
    <h4 style ="color:white;text-align:center;"> Know you have depression or not</h4>
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    Have_you_ever_had_suicidal_thoughts_ = st.text_input("Have you ever had suicidal thoughts (Yes or No)")
    Have_you_ever_had_suicidal_thoughts_.lower()
    if Have_you_ever_had_suicidal_thoughts_ == 'yes':
        Have_you_ever_had_suicidal_thoughts_ = 1
    elif Have_you_ever_had_suicidal_thoughts_ == 'no':
        Have_you_ever_had_suicidal_thoughts_ = 0
    Work_Study_Hours = (st.number_input("Work or Study Hours"))
    Financial_Stress = (st.number_input("Financial Stress(1-5)"))
    Pressure = (st.number_input("Pressure(1-5)"))
    
    
    result = ""
    
    if st.button("Predict"):
        result = prediction(Have_you_ever_had_suicidal_thoughts_,Work_Study_Hours,Financial_Stress,Pressure)
        if Work_Study_Hours > 10 and Financial_Stress >= 4 and Pressure >= 3:
            result = 1
        if result == 0:
            st.success('You have less chance of suffering from depression')
        else:
            st.success('You have high chance of suffering from depression')
            
            html_temp = """
            <div padding:8px"> 
            <h2>Measures you have to take for your mental health </h2>
            <ol>
                <li>Seek professional help from a therapist or counselor.</li>
                <li>Engage in regular physical exercise to improve mood and reduce stress.</li>
                <li>Practice mindfulness and relaxation techniques such as meditation or yoga.</li>
                <li>Ensure you get adequate sleep and maintain a regular sleep schedule.</li>
                <li>Reach out to friends and family for support and talk about your feelings.</li>
                <li>Consider reducing work or study hours if possible to manage stress levels.</li>
                <li>Develop a financial plan or seek financial counseling to address financial stress.</li>
                <li>Engage in hobbies and activities that you enjoy and that help you relax.</li>
                <li>Maintain a healthy diet and avoid excessive consumption of alcohol or drugs.</li>
                <li>Set realistic goals and break tasks into smaller, manageable steps.</li>
            </ol>
            </div> 
            """
            st.markdown(html_temp, unsafe_allow_html = True)
        
    # st.success('The output is {}'.format(result))
    

if __name__ == '__main__':
    main()