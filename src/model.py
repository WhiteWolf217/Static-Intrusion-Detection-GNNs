import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1=GCNConv(in_channels,hidden_channels)
        self.conv2=GCNConv(hidden_channels,hidden_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x

class EdgeClassifier(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_classes):
        super().__init__()
        self.encoder = GCNEncoder(node_feat_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x = self.encoder(data.x, data.edge_index)
        src, dst = data.edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        return self.classifier(edge_features)
    