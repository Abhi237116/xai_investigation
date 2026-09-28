from typing import Any

import networkx as nx
import pandas as pd


def build_graph(frame: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in frame.to_dict(orient="records"):
        account = f"account:{row['account_id']}"
        device = f"device:{row['device_id']}"
        beneficiary = f"beneficiary:{row['beneficiary_id']}"
        merchant = f"merchant:{row['merchant_id']}"
        transaction = f"transaction:{row['transaction_id']}"
        for node, kind, entity_id in [
            (account, "account", row["account_id"]),
            (device, "device", row["device_id"]),
            (beneficiary, "beneficiary", row["beneficiary_id"]),
            (merchant, "merchant", row["merchant_id"]),
            (transaction, "transaction", row["transaction_id"]),
        ]:
            graph.add_node(node, entity_type=kind, entity_id=str(entity_id))
        graph.add_edge(account, transaction, relation="initiated")
        graph.add_edge(account, device, relation="uses")
        graph.add_edge(account, beneficiary, relation="sends_to")
        graph.add_edge(account, merchant, relation="transacts_with")
    return graph


def neighborhood(graph: nx.Graph, node: str, depth: int = 2, limit: int = 100) -> dict[str, Any]:
    if depth < 1 or depth > 4:
        raise ValueError("depth must be between 1 and 4")
    if node not in graph:
        return {"nodes": [], "edges": [], "connected_accounts": [], "paths": []}
    distances = nx.single_source_shortest_path_length(graph, node, cutoff=depth)
    selected = set(sorted(distances, key=lambda value: (distances[value], value))[: limit + 1])
    edges = []
    for left, right, data in graph.subgraph(selected).edges(data=True):
        edges.append({"source": left, "target": right, "relation": data["relation"]})
    accounts = sorted(
        graph.nodes[item]["entity_id"]
        for item in selected
        if graph.nodes[item]["entity_type"] == "account" and item != node
    )
    return {
        "nodes": [
            {"id": item, **graph.nodes[item], "distance": distances[item]}
            for item in sorted(selected)
        ],
        "edges": edges,
        "connected_accounts": accounts,
        "paths": edges,
        "truncated": len(distances) > len(selected),
    }


def detect_shared_entities(graph: nx.Graph, account_id: str) -> list[dict[str, Any]]:
    account = f"account:{account_id}"
    if account not in graph:
        return []
    findings = []
    for entity in graph.neighbors(account):
        metadata = graph.nodes[entity]
        if metadata["entity_type"] not in {"device", "beneficiary", "merchant"}:
            continue
        related = sorted(
            graph.nodes[neighbor]["entity_id"]
            for neighbor in graph.neighbors(entity)
            if graph.nodes[neighbor]["entity_type"] == "account" and neighbor != account
        )
        if related:
            findings.append(
                {
                    "entity_type": metadata["entity_type"],
                    "entity_id": metadata["entity_id"],
                    "related_account_ids": related,
                    "category": "OBSERVED",
                    "evidence_source": "transaction_relationship_graph",
                }
            )
    return findings
