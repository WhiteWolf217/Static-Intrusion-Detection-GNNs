import torch
import pandas as pd
import numpy as np
import joblib
import os
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler=joblib.load('results/scalar.pkl')
node2id=joblib.load('results/node2id.pkl')
label_encoder=joblib.load('results/label_encoder.pkl')

existing_data=torch.load('processed_graphs/portscan_graph.pt')

new_file='data/ML/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df=pd.read_csv(new_file)
df.columns=df.columns.str.strip()

feature_cols = [
    'Flow Duration', 
    'Total Fwd Packets', 
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s'
]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=feature_cols, inplace=True)
df['src_node']=df['Source IP']+':'+df['Source Port'].astype(str)
df['dst_node'] = df['Destination IP'] + ':' + df['Destination Port'].astype(str)
df=df.reset_index(drop=True)

current_max=max(node2id.values())+1
new_nodes=set(df['src_node']).union(set(df['dst_node']))-set(node2id.keys())
node2id.update({node:current_max+i for i,node in enumerate(new_nodes)})

src_ids=df['src_node'].map(node2id)
dst_ids=df['dst_node'].map(node2id)
edge_index=torch.tensor([src_ids.values,dst_ids.values], dtype=torch.long)

features=scaler.transform(df[feature_cols])
edge_attr=torch.tensor(features,dtype=torch.float)

x_new=torch.zeros((len(node2id),len(feature_cols)))
node_counts=torch.zeros(len(node2id))

for i,row in df.iterrows():
    src=node2id[row['src_node']]
    dst=node2id[row['dst_node']]
    feats=torch.tensor(features[i], dtype=torch.float)
    x_new[src]+=feats
    x_new[dst]+=feats
    node_counts[src]+=1
    node_counts[dst]+=1

valid_nodes=node_counts>0
x_new[valid_nodes]=x_new[valid_nodes]/node_counts[valid_nodes].unsqueeze(1)

labels=label_encoder.transform(df['Label'])
edge_label=torch.tensor(labels,dtype=torch.long)

data=Data(
    x=x_new,
    edge_index=torch.cat([existing_data.edge_index,edge_index],dim=1),
    edge_attr=torch.cat([existing_data.edge_attr,edge_attr],dim=0),
    edge_label=torch.cat([existing_data.edge_label,edge_label],dim=0)
)

total_edges=data.edge_index.size(1)
train_mask=torch.zeros(total_edges,dtype=torch.bool)
test_mask=torch.zeros(total_edges,dtype=torch.bool)
train_mask[:int(0.8*total_edges)]=True
test_mask[int(0.8*total_edges):]=True

data.train_edge_mask=train_mask
data.test_edge_mask=test_mask

torch.save(data,'processed_graphs/portscan_graph.pt')
joblib.dump(node2id,'results/node2id.pkl')
print('Graph Updated')