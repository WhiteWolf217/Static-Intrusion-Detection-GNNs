# 🛡️ Static-Intrusion-Detection-GNNs v1.0

This project processes pre-labeled network traffic data (from CICIDS2017) in CSV format, constructs graphs where IPs and ports are treated as nodes, and uses a Graph Neural Network (GNN) to classify edges (flows) as either normal or malicious. The model is trained offline, evaluated using standard metrics, and used to predict on new static datasets.

# 📦 Prerequisites & Installation
    System Requirements
      Python 3.10
    
# 🚀 Features
    1. Direct file uploads (PCAP/CSV)
    2. Real-time attack visualization
    3. Downloadable reports (CSV)
    4. Interactive results dashboard

# ⚡ Quick Start
    1. Python Packages (installed via requirements.txt)
          pip install -r requirements.txt
    2. Run the app:
          streamlit run app.py
    3. Usage:
        1. Upload PCAP/CSV files via the sidebar
        2. View detected attacks in the interactive table
        3. Download results or save visualizations
# OR
    You can use my docker image to run it.
    To Install:
      docker pull whitewolf217/static-gnn-intrusion
    To Run:
      docker run -p 8501:8501 static-gnn-intrusion

# 📌 How to Train with Your Own Data
    To train the model using your own dataset, follow these steps:
    
    1. Prepare your data: First, place the path to your training data within the data_processing_incremental.py script.
    
    2. Process the data: Run the data processing script:
        'python data_processing_incremental.py'
    
    3. Train the model: After the data is processed, initiate training:
        'python train_incremental.py'
    
    4. Launch the application: Finally, run the Streamlit application to see your changes:
        'streamlit run app.py'
