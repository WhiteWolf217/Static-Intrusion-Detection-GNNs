# 🛡️ Static-Intrusion-Detection-GNNs


# 📦 Prerequisites & Installation
  System Requirements
    Python 3.10
    
# 🚀 Features
  1. Direct file uploads (PCAP/CSV)
  2. Real-time attack visualization
  3. Downloadable reports (CSV)
  4. Interactive results dashboard

# ⚡ Quick Start
  1. Install dependencies:
    pip install streamlit torch torch-geometric scapy pandas
  2. Run the app:
    streamlit run app.py
  3. Usage:
      1. Upload PCAP/CSV files via the sidebar
      2. View detected attacks in the interactive table
      3. Download results or save visualizations
# OR
  You can use my docker image to run it.
  
  link for docker image: 

# 📌 How to Train with Your Own Data
  To train the model using your own dataset, follow these steps:
  
  1. Prepare your data: First, place the path to your training data within the data_processing_incremental.py script.
  
  2. Process the data: Run the data processing script:
      'python data_processing_incremental.py'
  
  3. Train the model: After the data is processed, initiate training:
      'python train_incremental.py'
  
  4. Launch the application: Finally, run the Streamlit application to see your changes:
      'streamlit run app.py'
