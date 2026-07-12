import pickle
import random

# Load subgraph
try:
    with open("data/processed/subgraph.gpickle", "rb") as f:
        data = pickle.load(f)
        G = data["graph"]
except FileNotFoundError:
    print("Graph not found")
    exit()

n = G.number_of_nodes()
print(f"Nodes: {n}, Edges: {G.number_of_edges()}")

# Mock combined score (degree only for simplicity)
degrees = dict(G.degree())
sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)

k = int(0.15 * n)
patient_zeros = set(sorted_nodes[:k])

print(f"Patient zeros: {k}")

S_nodes = [node for node in G.nodes() if node not in patient_zeros]

# Simulate 1 step thresholds
random.seed(42)

threshold_crossed = 0
for node in S_nodes:
    neighbors = list(G.neighbors(node))
    if not neighbors:
        continue
    inf_neighbors = sum(1 for nb in neighbors if nb in patient_zeros)
    fraction_I = inf_neighbors / len(neighbors)
    thresh = max(0.05, min(0.95, random.gauss(0.3, 0.1)))
    if fraction_I >= thresh:
        threshold_crossed += 1

print(f"Susceptible nodes crossing the threshold at step 1: {threshold_crossed} / {len(S_nodes)}")
