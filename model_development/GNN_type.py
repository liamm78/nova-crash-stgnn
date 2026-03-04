import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset = Planetoid(root='data/Planetoid', name='Cora')

class CustomGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim ):
        super(CustomGNN, self).__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, output_dim)
        
    # Forward pass
    def forward(self, feature_data, edge_info):
        # First GCN Layer
        x = self.layer1(feature_data, edge_info)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Second GCN layer
        x = self.layer2(x, edge_info)
        # Softmax layer
        return F.log_softmax(x, dim=1)

# Initialize the model now. 
input_features = dataset.num_node_features  # nodes
num_classes = dataset.num_classes
print(input_features, num_classes)
model = CustomGNN(input_features, 16, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

graph_data = dataset[0] 
G = to_networkx(graph_data, to_undirected=True)

print(f"Nodes in NetworkX graph: {G.number_of_nodes()}")
print(f"Edges in NetworkX graph: {G.number_of_edges()}")

print(graph_data)







# Visualizing the citation graph


plt.figure(figsize=(15, 15))
# force directed algorithim for better viewing
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Draw edges first 
nx.draw_networkx_edges(G, pos, width=0.4, alpha=1, edge_color='gray')

# Draw nodes next
nx.draw_networkx_nodes(
    G, 
    pos, 
    node_size=20, 
    node_color=graph_data.y.numpy(), 
    cmap=plt.cm.jet, 
    alpha=0.8
)

plt.title("Full Planetoid Network")
plt.axis('off')
plt.show()