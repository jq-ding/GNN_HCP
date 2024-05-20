import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import os
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_add_pool
from torch.nn import Linear
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import random_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GIN', 'GAT'])
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    return args

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(gpu_id):
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

def load_data():
    train_dataset = torch.load('./data/LR_dataset.pth')
    test_dataset = torch.load('./data/RL_dataset.pth')
    # train_dataset = Dataset_graph(./data/LR_graph')
    # test_dataset = Dataset_graph('./data/RL_graph')
    val_set, test_set = random_split(test_dataset, [2972, 4458])
    
    batchsize = 64
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batchsize, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=True, num_workers=8)
    
    return train_loader, val_loader, test_loader

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.lin1 = Linear(hidden_size, in_feats)
        self.lin2 = Linear(in_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, data.batch)
        return self.mlp(x)
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

def get_model(model_name, hidden_channels):
    if model_name == 'GCN':
        return GCN(in_feats=360, hidden_size=hidden_channels, out_feats=7)
    elif model_name == 'GIN':
        return GIN(in_channels=360, hidden_channels=hidden_channels, out_channels=7, num_layers=2)
    elif model_name == 'GAT':
        return GAT(in_channels=360, hidden_channels=hidden_channels, out_channels=7, heads=2)
    else:
        raise argparse.ArgumentTypeError(f"Unsupported model. Choose from GCN, GIN, GAT.")

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    correct = 0
    preds = []
    gts = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=-1)
            correct += int((pred == data.y).sum())
            preds.append(pred.cpu().numpy())
            gts.append(data.y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    accuracy = accuracy_score(gts, preds)
    precision = precision_score(gts, preds, average='weighted')
    recall = recall_score(gts, preds, average='weighted')
    f1 = f1_score(gts, preds, average='weighted')
    return accuracy, precision, recall, f1

def main():
    args = parse_args()
    set_seeds()
    device = get_device(args.gpu)
    train_loader, val_loader, test_loader = load_data()
    model = get_model(args.model, args.hidden_channels).to(device)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, parameters: {num_parameters}")
    
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, criterion, optimizer, device)
        val_acc, _, _, _ = test(model, val_loader, device)
        test_acc, pre, rec, f1 = test(model, test_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f'Epoch: {epoch:03d}, best Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}, Loss: {loss:.4f}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}')

if __name__ == "__main__":
    main()
