import streamlit as st
import requests


def main():

    st.title("Agro-Chem Model API")
    message = st.text_input('Text_Query')

    if st.button('Predict'):
        payload = {
            "text": message
        }
        res = requests.post(f"http://127.0.0.1:8000/predict-Label",json=payload )
        with st.spinner('Classifying, please wait....'):
            st.write(res.json())




if __name__ == '__main__':
    main()