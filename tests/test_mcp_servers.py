import json
import importlib
import pytest
from fastmcp import Client
from src.schemas.axiom_rules import SILENCING_THRESHOLD


@pytest.mark.asyncio
async def test_genomics_get_mutations_returns_braf(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)

    async with Client(genomics_server.mcp) as client:
        result = await client.call_tool("get_mutations", {"cell_line": "A375"})
        data = json.loads(result.data)
    mutations = data["mutations"]
    assert len(mutations) == 1
    assert mutations[0]["gene"] == "BRAF"
    assert mutations[0]["mutation"] == "V600E"


@pytest.mark.asyncio
async def test_genomics_get_mutations_filtered_by_gene(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)

    async with Client(genomics_server.mcp) as client:
        result = await client.call_tool("get_mutations", {"cell_line": "A375", "gene": "BRAF"})
        data = json.loads(result.data)
    mutations = data["mutations"]
    assert len(mutations) == 1
    assert mutations[0]["gene"] == "BRAF"


@pytest.mark.asyncio
async def test_genomics_get_mutations_drug_lookup_uses_mutation_column(genomics_db, monkeypatch):
    # Production genomics.db has no `protein_change` column — the mutations table
    # only has `mutation`. get_mutations must derive the CIViC lookup variant from
    # `mutation`, not from a `protein_change` key that doesn't exist in real data
    # (the test fixture's `protein_change='p.V600E'` is a stale artifact of a
    # schema that was never actually shipped — real cell lines only have
    # `mutation='V600E'`).
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)

    captured = {}

    def fake_lookup_civic(gene, protein_change, drug):
        captured["protein_change"] = protein_change
        return []

    monkeypatch.setattr(genomics_server, "_lookup_civic", fake_lookup_civic)

    async with Client(genomics_server.mcp) as client:
        await client.call_tool("get_mutations", {"cell_line": "A375", "drug": "Vemurafenib"})

    assert captured["protein_change"] == "V600E"


@pytest.mark.asyncio
async def test_genomics_get_mutations_unknown_cell_line_returns_empty(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)

    async with Client(genomics_server.mcp) as client:
        result = await client.call_tool("get_mutations", {"cell_line": "UNKNOWN"})
        data = json.loads(result.data)
    assert data["mutations"] == []
    assert data["cell_line_in_db"] is False


@pytest.mark.asyncio
async def test_genomics_get_cnv_returns_braf(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)

    async with Client(genomics_server.mcp) as client:
        result = await client.call_tool("get_cnv", {"cell_line": "A375"})
        cnv = json.loads(result.data)
    assert len(cnv) >= 1
    assert cnv[0]["gene"] == "BRAF"
    assert cnv[0]["status"] == "neutral"


@pytest.mark.asyncio
async def test_genomics_check_mutation_impact_driver(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)

    async with Client(genomics_server.mcp) as client:
        result = await client.call_tool("check_mutation_impact", {"gene": "BRAF", "mutation": "V600E"})
        impact = json.loads(result.data)
    assert impact["is_driver"] is True
    assert "A375" in impact["cell_lines"]


# --- Transcriptomics MCP ---

@pytest.mark.asyncio
async def test_transcriptomics_get_expression(transcriptomics_db, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOMICS_DB", str(transcriptomics_db))
    from src.mcp_servers import transcriptomics_server
    importlib.reload(transcriptomics_server)

    async with Client(transcriptomics_server.mcp) as client:
        result = await client.call_tool("get_expression", {"cell_line": "A375", "gene": "BRAF"})
        data = json.loads(result.data)
    assert len(data) == 1
    assert data[0]["gene"] == "BRAF"
    assert data[0]["tpm"] == pytest.approx(45.2, rel=1e-2)


@pytest.mark.asyncio
async def test_transcriptomics_check_silencing_true(transcriptomics_db, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOMICS_DB", str(transcriptomics_db))
    from src.mcp_servers import transcriptomics_server
    importlib.reload(transcriptomics_server)

    async with Client(transcriptomics_server.mcp) as client:
        result = await client.call_tool("check_silencing", {"cell_line": "A375", "gene": "EGFR"})
        data = json.loads(result.data)
    assert data["is_silenced"] is True   # z=-2.3 < SILENCING_THRESHOLD (-2.0)
    assert data["z_score"] == pytest.approx(-2.3)
    assert data["threshold_used"] == SILENCING_THRESHOLD


@pytest.mark.asyncio
async def test_transcriptomics_check_silencing_false(transcriptomics_db, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOMICS_DB", str(transcriptomics_db))
    from src.mcp_servers import transcriptomics_server
    importlib.reload(transcriptomics_server)

    async with Client(transcriptomics_server.mcp) as client:
        result = await client.call_tool("check_silencing", {"cell_line": "A375", "gene": "BRAF"})
        data = json.loads(result.data)
    assert data["is_silenced"] is False  # z=1.3 > SILENCING_THRESHOLD


@pytest.mark.asyncio
async def test_transcriptomics_check_silencing_missing_gene(transcriptomics_db, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOMICS_DB", str(transcriptomics_db))
    from src.mcp_servers import transcriptomics_server
    importlib.reload(transcriptomics_server)

    async with Client(transcriptomics_server.mcp) as client:
        result = await client.call_tool("check_silencing", {"cell_line": "A375", "gene": "UNKNOWN"})
        data = json.loads(result.data)
    assert data["is_silenced"] is None
    assert "reason" in data


# --- Pharmacology MCP ---

@pytest.mark.asyncio
async def test_pharmacology_get_ic50_returns_response(pharmacology_db, monkeypatch):
    monkeypatch.setenv("PHARMACOLOGY_DB", str(pharmacology_db))
    from src.mcp_servers import pharmacology_server
    importlib.reload(pharmacology_server)

    async with Client(pharmacology_server.mcp) as client:
        result = await client.call_tool("get_ic50", {"cell_line": "A375", "drug": "Vemurafenib"})
        data = json.loads(result.data)
    assert data["ln_ic50"] == pytest.approx(-1.2)
    assert data["z_score"] == pytest.approx(-1.5)
    assert data["label"] == "SENSITIVE"


@pytest.mark.asyncio
async def test_pharmacology_get_ic50_missing_returns_error(pharmacology_db, monkeypatch):
    monkeypatch.setenv("PHARMACOLOGY_DB", str(pharmacology_db))
    from src.mcp_servers import pharmacology_server
    importlib.reload(pharmacology_server)

    async with Client(pharmacology_server.mcp) as client:
        result = await client.call_tool("get_ic50", {"cell_line": "A375", "drug": "UNKNOWN"})
        data = json.loads(result.data)
    assert "error" in data


@pytest.mark.asyncio
async def test_pharmacology_get_drug_info(pharmacology_db, monkeypatch):
    monkeypatch.setenv("PHARMACOLOGY_DB", str(pharmacology_db))
    from src.mcp_servers import pharmacology_server
    importlib.reload(pharmacology_server)

    async with Client(pharmacology_server.mcp) as client:
        result = await client.call_tool("get_drug_info", {"drug": "Vemurafenib"})
        data = json.loads(result.data)
    assert data["target_genes"] == "BRAF"
    assert data["pathway"] == "MAPK"


@pytest.mark.asyncio
async def test_pharmacology_get_sensitivity_profile(pharmacology_db, monkeypatch):
    monkeypatch.setenv("PHARMACOLOGY_DB", str(pharmacology_db))
    from src.mcp_servers import pharmacology_server
    importlib.reload(pharmacology_server)

    async with Client(pharmacology_server.mcp) as client:
        result = await client.call_tool("get_sensitivity_profile", {"drug": "Vemurafenib"})
        data = json.loads(result.data)
    assert len(data) == 1
    assert data[0]["mutation"] == "V600E"
    assert data[0]["resistant_fraction"] == pytest.approx(0.1)


# --- Pathway MCP ---

@pytest.mark.asyncio
async def test_pathway_get_pathway_genes(pathways_db, monkeypatch):
    monkeypatch.setenv("PATHWAYS_DB", str(pathways_db))
    from src.mcp_servers import pathway_server
    importlib.reload(pathway_server)

    async with Client(pathway_server.mcp) as client:
        result = await client.call_tool("get_pathway_genes", {"pathway_id": "hsa04010"})
        data = json.loads(result.data)
    assert len(data) >= 1
    genes = [r["gene"] for r in data]
    assert "BRAF" in genes


@pytest.mark.asyncio
async def test_pathway_check_bypass_exists(pathways_db, monkeypatch):
    monkeypatch.setenv("PATHWAYS_DB", str(pathways_db))
    from src.mcp_servers import pathway_server
    importlib.reload(pathway_server)

    async with Client(pathway_server.mcp) as client:
        result = await client.call_tool("check_bypass", {"pathway_id": "hsa04010", "blocked_gene": "BRAF"})
        data = json.loads(result.data)
    assert data["bypass_exists"] is True
    assert "MEK2" in data["bypass_genes"]


@pytest.mark.asyncio
async def test_pathway_get_upstream_regulators(pathways_db, monkeypatch):
    monkeypatch.setenv("PATHWAYS_DB", str(pathways_db))
    from src.mcp_servers import pathway_server
    importlib.reload(pathway_server)

    async with Client(pathway_server.mcp) as client:
        result = await client.call_tool("get_upstream_regulators", {"gene": "BRAF"})
        data = json.loads(result.data)
    assert len(data) >= 1
    regulators = [r["regulator"] for r in data]
    assert "RAS" in regulators
