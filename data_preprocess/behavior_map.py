import re
import pydot
import networkx as nx
from pydot import Node as PydotNode, Edge as PydotEdge, Graph as PydotGraph

def extract_json_ids_from_label(label: str) -> list[str]:
    return re.findall(r'(?:^|\t|\\l)([0-9_]+\.json)', label)

def recursively_add_nodes_edges(G: nx.DiGraph, graph_or_subgraph: PydotGraph):
    # 加入節點
    for p_node in graph_or_subgraph.get_nodes():
        name = p_node.get_name()
        if not name or name in {"node", "edge", "graph"}:
            continue

        node_id = name.strip('"')
        label = p_node.get_attributes().get("label", "").strip('"')
        json_ids = extract_json_ids_from_label(label)
        G.add_node(node_id, label=label, rules=json_ids)

    # 加入邊
    for edge in graph_or_subgraph.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        attrs = edge.get_attributes()
        G.add_edge(src, dst, **attrs)

    # 遞迴處理 subgraph
    for subgraph in graph_or_subgraph.get_subgraphs():
        recursively_add_nodes_edges(G, subgraph)

def load_dot_to_networkx(dot_path: str) -> nx.DiGraph:
    graphs = pydot.graph_from_dot_file(dot_path)
    if not graphs:
        raise ValueError("無法從 dot 檔案載入圖")
    
    G = nx.DiGraph()
    recursively_add_nodes_edges(G, graphs[0])
    return G
