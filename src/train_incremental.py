import torch
import torch.nn as nn
import os
import joblib
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from model import EdgeClassifier

data = torch.load('processed_graphs/portscan_graph.pt', weights_only=False)
scaler= joblib.load('results/scalar.pkl')
node2id=joblib.load('results/node2id.pkl')
label_encoder=joblib.load('results/label_encoder.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

num_node_features=7

model=EdgeClassifier(
    node_feat_dim=num_node_features,
    hidden_dim=64,
    num_classes=len(label_encoder.classes_)
).to(device)

model.load_state_dict(torch.load('results/edge_classifier.pt',map_location=device))

classes = torch.unique(data.edge_label.cpu())
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes.numpy(),
    y=data.edge_label.cpu().numpy()
)

class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)

criterion=nn.CrossEntropyLoss(weight=class_weights)
optimizer=torch.optim.Adam(model.parameters(),lr=0.005)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_edge_mask], data.edge_label[data.train_edge_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(data)
        preds = out.argmax(dim=1)
        correct = (preds[data.test_edge_mask] == data.edge_label[data.test_edge_mask]).sum()
        acc = int(correct) / int(data.test_edge_mask.sum())
    return acc, preds

losses = []
for epoch in range(1, 101):
    loss = train()
    losses.append(loss)
    if epoch % 10 == 0:
        acc, _ = test()
        print(f"Epoch: {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

plt.plot(losses)
plt.title("Incremental Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

os.makedirs('results',exist_ok=True)
torch.save(model.state_dict(),'results/edge_classifier.pt')
print('Incremental Done')
