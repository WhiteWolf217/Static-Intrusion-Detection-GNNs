import torch
import torch.nn as nn
import os
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from model import EdgeClassifier

data = torch.load('processed_graphs/portscan_graph.pt', weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

model = EdgeClassifier(
    node_feat_dim=data.num_node_features,
    hidden_dim=64,
    num_classes=len(torch.unique(data.edge_label))
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(data.edge_label.cpu().numpy()),
    y=data.edge_label.cpu().numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

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
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

_, preds = test()
print("\nClassification Report:")
print(classification_report(data.edge_label[data.test_edge_mask].cpu(), preds[data.test_edge_mask].cpu()))

os.makedirs('results', exist_ok=True)
torch.save(model.state_dict(), 'results/edge_classifier.pt')    