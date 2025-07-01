import torch
import pandas as pd
import numpy as np
import joblib as jl
import sys
import os
from model import EdgeClassifier
from torch_geometric.data import Data
from pcap_processor import pcap_to_flows

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results/predictions', exist_ok=True)

scalar=jl.load('results/scalar.pkl')
node2id=jl.load('results/node2id.pkl')
le = jl.load('results/label_encoder.pkl')

model=EdgeClassifier(node_feat_dim=7,hidden_dim=64,num_classes=15).to(device)
model.load_state_dict(torch.load('results/edge_classifier.pt',map_location=device))
model.eval()

feature_cols=[
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s'
]

def prepare_graph_data(df):
    df.columns = df.columns.str.strip()
    df['src_node']=df['Source IP']+':'+df['Source Port'].astype(str)
    df['dst_node']=df['Destination IP']+':'+df['Destination Port'].astype(str)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    current_max_id=max(node2id.values()) if node2id else 0
    new_nodes=set(df['src_node']).union(set(df['dst_node']))-set(node2id.keys())
    node2id.update({node:current_max_id + i +1 for i,node in enumerate(new_nodes)})

    edge_index=torch.tensor([df['src_node'].map(node2id), df['dst_node'].map(node2id)],dtype=torch.long)
    edge_attr=torch.tensor(scalar.transform(df[feature_cols]),dtype=torch.float)
    return Data(
        x=torch.zeros((len(node2id), len(feature_cols)),dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr
    ).to(device),df

def predict(data):
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confs = torch.max(probs, dim=1).values.cpu().numpy()
        return preds, confs

def predict_pcap(pcap_path):
    df=pcap_to_flows(pcap_path)
    data, df=prepare_graph_data(df)
    preds, confs= predict(data)
    decoded_preds = le.inverse_transform(preds)
    return format_results(df,decoded_preds,confs)

def predict_flows(csv_path):
    df=pd.read_csv(csv_path)
    data, df=prepare_graph_data(df)
    preds,confs=predict(data)
    decoded_preds = le.inverse_transform(preds)
    return format_results(df,decoded_preds,confs)

def format_results(df, preds, confs):
    return pd.DataFrame([{
        'source_ip': row['Source IP'],
        'source_port': row['Source Port'],
        'dest_ip': row['Destination IP'],
        'dest_port': row['Destination Port'],
        'prediction': preds[i],
        'confidence': confs[i],
        'flow_duration': row['Flow Duration'],
        'total_packets': row['Total Fwd Packets'] + row['Total Backward Packets']
    } for i, row in df.iterrows()])

if __name__=='__main__':
    if len(sys.argv)<2:
        print('Usage: python predict.py <path to file>')
        sys.exit(1)

    input_path=sys.argv[1]
    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else f'results/predictions/{os.path.basename(input_path).split(".")[0]}_predictions.csv'
    )

    try:
        if input_path.endswith('.pcap'):
            results=predict_pcap(input_path)
        elif input_path.endswith('.csv'):
            results=predict_flows(input_path)
        else:
            raise ValueError('Unsupported file format')
        
        results.to_csv(output_path,index=False)
        jl.dump(node2id,'results/node2id.pkl')
        print(f'Saved predictions to {output_path}')
        
    except Exception as e:
        print(f'Error:{str(e)}')
        sys.exit(1)