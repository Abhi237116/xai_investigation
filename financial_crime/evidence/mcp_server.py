import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from financial_crime.evidence import tool_service
from financial_crime.evidence.tool_service import TOOL_MANIFEST

mcp = FastMCP("Financial Crime Investigation Tools")


@mcp.tool()
def get_transaction(transaction_id: str) -> dict[str, Any]:
    """READ-ONLY (10s). Retrieve a transaction; returns structured evidence or not-found status."""
    return tool_service.get_transaction(transaction_id)


@mcp.tool()
def get_account_history(account_id: str, limit: int = 50) -> dict[str, Any]:
    """READ-ONLY (10s). Retrieve up to 200 account transactions; invalid limits are rejected."""
    return tool_service.get_account_history(account_id, limit)


@mcp.tool()
def get_device_history(device_id: str, limit: int = 50) -> dict[str, Any]:
    """READ-ONLY (10s). Retrieve up to 200 transactions for a device."""
    return tool_service.get_device_history(device_id, limit)


@mcp.tool()
def get_beneficiary_history(beneficiary_id: str, limit: int = 50) -> dict[str, Any]:
    """READ-ONLY (10s). Retrieve up to 200 transactions for a beneficiary."""
    return tool_service.get_beneficiary_history(beneficiary_id, limit)


@mcp.tool()
def find_related_accounts(account_id: str, depth: int = 2) -> dict[str, Any]:
    """READ-ONLY (10s). Find connected accounts and observed shared entities."""
    return tool_service.find_related_accounts(account_id, depth)


@mcp.tool()
def get_network_connections(entity_type: str, entity_id: str, depth: int = 2) -> dict[str, Any]:
    """READ-ONLY (10s). Return bounded graph nodes and relationship edges."""
    return tool_service.get_network_connections(entity_type, entity_id, depth)


@mcp.tool()
def get_model_prediction(transaction_id: str) -> dict[str, Any]:
    """READ-ONLY (10s). Return the trained model score or explicit unavailable status."""
    return tool_service.get_model_prediction(transaction_id)


@mcp.tool()
def get_shap_explanation(transaction_id: str) -> dict[str, Any]:
    """READ-ONLY (10s). Return model-generated SHAP values or explanation_unavailable."""
    return tool_service.get_shap_explanation(transaction_id)


@mcp.tool()
def search_evidence(case_id: str, query: str, limit: int = 25) -> dict[str, Any]:
    """READ-ONLY (10s). Search stored, provenance-labeled evidence for a case."""
    return tool_service.search_evidence(case_id, query, limit)


@mcp.tool()
def generate_counterfactual(transaction_id: str) -> dict[str, Any]:
    """READ-ONLY (10s). Simulate feature substitutions without predicting real-world prevention."""
    return tool_service.generate_counterfactual_for_transaction(transaction_id)


@mcp.resource("investigation://tool-manifest")
def tool_manifest() -> str:
    """Tool names, JSON schemas, permission levels, timeouts, and error behavior."""
    return json.dumps(TOOL_MANIFEST)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
