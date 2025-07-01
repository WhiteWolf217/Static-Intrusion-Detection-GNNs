import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import joblib

os.makedirs('processed_graphs', exist_ok=True)
os.makedirs('results',exist_ok=True)
folder='data/ML'
all_files=[os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

dfs=[]
for file in all_files:
    print(f"Loading {file}")
    try:
        df=pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"Unicode error in {file}")
        df=pd.read_csv(file, encoding='ISO-8859-1')
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()

print(df.columns)
print(df.head())

feature_cols = [
    'Flow Duration', 
    'Total Fwd Packets', 
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s'
]

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

df['src_node'] = df['Source IP'] + ':' + df['Source Port'].astype(str)
df['dst_node'] = df['Destination IP'] + ':' + df['Destination Port'].astype(str)

df = df.reset_index(drop=True)

all_nodes = pd.unique(df[['src_node', 'dst_node']].values.ravel())
node2id = {node: idx for idx, node in enumerate(all_nodes)}

src_ids = df['src_node'].map(node2id)
dst_ids = df['dst_node'].map(node2id)
edge_array = np.array([df['src_node'].map(node2id).values, df['dst_node'].map(node2id).values])
edge_index = torch.tensor(edge_array, dtype=torch.long)

scaler = StandardScaler()
features = scaler.fit_transform(df[feature_cols])

x = torch.zeros((len(node2id), len(feature_cols)), dtype=torch.float)
node_counts = torch.zeros(len(node2id))

for i, row in df.iterrows():
    src = node2id[row['src_node']]
    dst = node2id[row['dst_node']]
    current_features = torch.tensor(features[i], dtype=torch.float)
    
    x[src] += current_features
    x[dst] += current_features
    node_counts[src] += 1
    node_counts[dst] += 1

valid_nodes = node_counts > 0
x[valid_nodes] = x[valid_nodes] / node_counts[valid_nodes].unsqueeze(1)

le = LabelEncoder()
labels = le.fit_transform(df['Label'])

num_edges = edge_index.size(1)
indices = torch.randperm(num_edges)
train_count = int(0.8 * num_edges)

train_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
test_edge_mask = torch.zeros(num_edges, dtype=torch.bool)

train_edge_mask[indices[:train_count]] = True
test_edge_mask[indices[train_count:]] = True

edge_labels = torch.tensor(labels, dtype=torch.long)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=torch.tensor(features, dtype=torch.float),
    edge_label=edge_labels,
    train_edge_mask=train_edge_mask,
    test_edge_mask=test_edge_mask
)


torch.save(data, 'processed_graphs/portscan_graph.pt')
joblib.dump(scaler,'results/scalar.pkl')
joblib.dump(node2id,'results/node2id.pkl')
joblib.dump(le, 'results/label_encoder.pkl')

print("Label distribution:")
print(df['Label'].value_counts())

print("\nGraph saved with:")
print(f"- {data.num_nodes} nodes")
print(f"- {data.num_edges} edges")
print(f"- {data.num_node_features} node features")
print(f"- {len(le.classes_)} classes: {le.classes_}")
