import streamlit as st
import pandas as pd
import os
import tempfile
import subprocess

st.set_page_config(page_title="GNN Intrusion Detection",layout='centered')
st.title('🔐 GNN Intrusion Detection System')
st.markdown('Upload a network traffic file (.csv or .pcap)')
uploaded_file=st.file_uploader('choose file', type=['csv','pcap'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix=f'_{uploaded_file.name}') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path=tmp_file.name

    st.success(f'uploaded file saved to: {temp_path}')

    output_path='prediction_streamlit.csv'
    with st.spinner('Running prediction...'):
        try:
            result=subprocess.run(
                ['python','src/predict.py',temp_path,output_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                st.error(f'Prediction Falied: {result.stderr}')
            else:
                st.success('Prediction Completed')
                predictions=pd.read_csv(output_path)
                st.dataframe(predictions.head(10))

                with open(output_path,'rb') as f:
                    st.download_button('Download Prediction CSV', f, file_name='prediction.csv')
        except Exception as e:
            st.error(f'Unexpec  ted error: {str(e)}')