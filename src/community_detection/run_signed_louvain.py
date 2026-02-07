import os
from .signed_louvain import best_partition
from .utils import build_nx_graph, build_subgraphs


def run_signed_louvain(edge_list, nb_nodes, save_dir, seed=42):
    """
    Build a graph from the edge list, run Signed Louvain, and save the detected communities.
    
    Args:
        edge_list (list): List of edges in the format [(source, target, weight), ...].
        nb_nodes (int): Number of unique nodes in the graph.
        save_dir (str): Directory to save the community detection results.
    
    Returns:
        dict: Detected communities in the format {community_id: [node_list]}.
        Graph: NetworkX graph object.
    """
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

    # Build graph using build_nx_graph
    graph = build_nx_graph(nb_nodes, edge_list)

    # Build positive and negative subgraphs
    pos_graph, neg_graph = build_subgraphs(graph, weight='weight')

    # Run Signed Louvain
    communities_dict = best_partition(
        layers=[pos_graph, neg_graph],  # Two graphs passed
        resolutions=[1.0, 1.0],        # Resolution for each graph
        layer_weights=[1.0, -1.0],     # Positive/negative weights
        random_state=seed
    )

    # Reformat communities to {community_id: [node_list]}
    communities = {}
    for node, community_id in communities_dict.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    # Save communities to CSV in {community_id: [node_list]} format
    communities_path = os.path.join(save_dir, "communities.csv")
    communities_data = [{"community_id": key, "nodes": ",".join(map(str, value))}
                        for key, value in communities.items()]

    # Convert to DataFrame and sort by community_id
    import pandas as pd
    communities_df = pd.DataFrame(communities_data).sort_values(by="community_id")
    communities_df.to_csv(communities_path, index=False)
    print(f"Communities saved to {communities_path} (sorted by community_id)")

    return communities, graph
