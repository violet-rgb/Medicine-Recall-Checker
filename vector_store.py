"""
Chroma-backed vector store of known recalled medicine batches.
Seeded once from data/recalled_batches.json. Used as the first lookup layer
before falling back to a live web search via the LLM.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from decouple import config

CHROMA_DIR = config("CHROMA_DIR", default="./chroma_db")
DATA_FILE = Path(__file__).parent / "data" / "recalled_batches.json"
COLLECTION_NAME = "recalled_batches"
MATCH_THRESHOLD = config("VECTOR_MATCH_THRESHOLD", default=0.35, cast=float)
DATASET_HASH_KEY = "dataset_sha256"

_client = None
_collection = None
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _doc_text(entry: dict) -> str:
    entry = _normalize_entry(entry)
    return (
        f"{entry['medicine_name']} ({', '.join(entry['aliases'])}) - "
        f"Batch: {entry['batch_number']}, Manufacturer: {entry['manufacturer']}. "
        f"Status: {entry['recall_status']}. Reason: {entry['recall_reason']}. "
        f"Reporting agency: {entry['recalling_agency']}"
    )


def _clean_batch(batch_number: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(batch_number or "").upper())


def _clean_optional(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _format_dataset_date(value):
    value = _clean_optional(value)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.date().isoformat()


def _format_reporting_month(value):
    value = _clean_optional(value)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%B %Y")


def _data_file_hash() -> str:
    return hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()


def _is_nsq_entry(entry: dict) -> bool:
    return "Name of Product" in entry or "NSQ Result" in entry or "Batch No" in entry


def _normalize_entry(entry: dict) -> dict:
    if _is_nsq_entry(entry):
        product_name = _clean_optional(entry.get("Name of Product")) or ""
        batch_number = _clean_optional(entry.get("Batch No")) or ""
        manufacturer = _clean_optional(entry.get("Manufactured By")) or ""
        nsq_result = _clean_optional(entry.get("NSQ Result")) or "Not of standard quality"
        reporting_source = _clean_optional(entry.get("Reporting Source"))
        reporting_lab = _clean_optional(entry.get("Reporting by Lab/State"))
        reporting_month = _format_reporting_month(entry.get("Reporting Month & Year"))
        mfg_date = _format_dataset_date(entry.get("Manufacturing Date"))
        expiry_date = _format_dataset_date(entry.get("Expiry Date"))
        reporting_parts = [part for part in (reporting_source, reporting_lab) if part]

        return {
            "medicine_name": product_name,
            "batch_number": batch_number,
            "aliases": [product_name],
            "manufacturer": manufacturer,
            "recall_status": "NSQ",
            "is_recalled": True,
            "recall_date": reporting_month,
            "recall_reason": nsq_result,
            "recalling_agency": " - ".join(reporting_parts) or "NSQ Dataset",
            "recall_class": "NSQ (Not of Standard Quality)",
            "recommendation": (
                "Do not use this batch until it has been verified by a pharmacist, "
                "healthcare professional, or the relevant drug control authority."
            ),
            "mfg_date": mfg_date,
            "expiry_date": expiry_date,
            "reporting_source": reporting_source,
            "reporting_lab": reporting_lab,
            "record_id": _clean_optional(entry.get("S.No")),
        }

    recall_status = entry.get("recall_status") or (
        "NSQ" if entry.get("is_recalled") else "Not in NSQ"
    )
    is_recalled = bool(entry.get("is_recalled")) or str(recall_status).strip().lower() in {
        "recalled",
        "nsq",
        "nsq / recalled",
    }
    aliases = entry.get("aliases") or [entry.get("generic_name"), entry.get("composition")]

    return {
        "medicine_name": _clean_optional(entry.get("medicine_name")) or "",
        "batch_number": _clean_optional(entry.get("batch_number") or entry.get("batch_no")) or "",
        "aliases": [alias for alias in aliases if alias],
        "manufacturer": _clean_optional(entry.get("manufacturer")) or "",
        "recall_status": recall_status,
        "is_recalled": is_recalled,
        "recall_date": _clean_optional(entry.get("recall_date")),
        "recall_reason": _clean_optional(entry.get("recall_reason")),
        "recalling_agency": _clean_optional(entry.get("recalling_agency")) or "Internal Database",
        "recall_class": _clean_optional(entry.get("recall_class")),
        "recommendation": entry.get("recommendation")
        or (
            "Do not use this batch. Contact a pharmacist or healthcare professional."
            if is_recalled
            else "No recall action required based on the internal database."
        ),
        "mfg_date": _format_dataset_date(entry.get("mfg_date") or entry.get("manufacturing_date")),
        "expiry_date": _format_dataset_date(entry.get("expiry_date")),
        "reporting_source": None,
        "reporting_lab": None,
        "record_id": _clean_optional(entry.get("id") or entry.get("record_id")),
    }


def _load_json_entries() -> list[dict]:
    with open(DATA_FILE, "r") as f:
        return [_normalize_entry(entry) for entry in json.load(f)]


def init_vector_store(force_reseed: bool = False):
    """Creates the collection and seeds it from recalled_batches.json when needed."""
    global _collection
    client = _get_client()

    if force_reseed:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    dataset_hash = _data_file_hash()
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedder,
        metadata={DATASET_HASH_KEY: dataset_hash},
    )

    entries = _load_json_entries()
    collection_hash = (_collection.metadata or {}).get(DATASET_HASH_KEY)
    should_seed = (
        _collection.count() == 0
        or _collection.count() != len(entries)
        or collection_hash != dataset_hash
    )

    if should_seed:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedder,
            metadata={DATASET_HASH_KEY: dataset_hash},
        )

        _collection.add(
            ids=[str(i) for i in range(len(entries))],
            documents=[_doc_text(e) for e in entries],
            metadatas=[
                {
                    "medicine_name": e["medicine_name"],
                    "batch_number": e["batch_number"],
                    "normalized_batch": _clean_batch(e["batch_number"]),
                    "aliases": json.dumps(e["aliases"]),
                    "manufacturer": e["manufacturer"],
                    "recall_status": e["recall_status"],
                    "is_recalled": e["is_recalled"],
                    "recall_date": e["recall_date"] or "",
                    "recall_reason": e["recall_reason"] or "",
                    "recalling_agency": e["recalling_agency"],
                    "recall_class": e["recall_class"] or "",
                    "recommendation": e["recommendation"],
                    "mfg_date": e["mfg_date"] or "",
                    "expiry_date": e["expiry_date"] or "",
                    "reporting_source": e["reporting_source"] or "",
                    "reporting_lab": e["reporting_lab"] or "",
                    "record_id": e["record_id"] or "",
                }
                for e in entries
            ],
        )
    return _collection


def get_collection():
    global _collection
    if _collection is None:
        init_vector_store()
    return _collection


def search_recalled_db(batch_number: str, medicine_name: str = "", top_k: int = 3) -> list[dict]:
    """
    Searches the recalled batches database.
    First tries direct exact match (case-insensitive, whitespace stripped) on batch_number.
    If no exact match, falls back to semantic vector search.
    """
    # 1. Try authoritative JSON exact batch matching first.
    clean_batch = _clean_batch(batch_number)
    if clean_batch:
        exact_matches = [
            entry for entry in _load_json_entries()
            if _clean_batch(entry["batch_number"]) == clean_batch
        ]
        if exact_matches:
            return [{**entry, "distance": 0.0} for entry in exact_matches]

    collection = get_collection()

    # 2. Try exact batch matching against Chroma metadata for older seeded stores.
    if clean_batch:
        try:
            get_results = collection.get(where={"normalized_batch": clean_batch})
            if get_results and get_results["ids"]:
                matches = []
                for i in range(len(get_results["ids"])):
                    meta = get_results["metadatas"][i]
                    matches.append({
                        "medicine_name": meta["medicine_name"],
                        "batch_number": meta["batch_number"],
                        "aliases": json.loads(meta["aliases"]),
                        "manufacturer": meta["manufacturer"],
                        "recall_status": meta.get("recall_status", "Recalled"),
                        "is_recalled": meta.get("is_recalled", True),
                        "recall_date": meta["recall_date"],
                        "recall_reason": meta["recall_reason"],
                        "recalling_agency": meta["recalling_agency"],
                        "recall_class": meta["recall_class"],
                        "recommendation": meta["recommendation"],
                        "mfg_date": meta.get("mfg_date", ""),
                        "expiry_date": meta.get("expiry_date", ""),
                        "reporting_source": meta.get("reporting_source", ""),
                        "reporting_lab": meta.get("reporting_lab", ""),
                        "record_id": meta.get("record_id", ""),
                        "distance": 0.0  # Exact match
                    })
                return matches
        except Exception as e:
            print(f"[vector_store] Error in exact batch match: {e}")

    # 3. Semantic query fallback
    query_parts = []
    if medicine_name:
        query_parts.append(medicine_name)
    if batch_number:
        query_parts.append(f"batch {batch_number}")
    query_text = " ".join(query_parts).strip()
    
    if not query_text:
        return []

    results = collection.query(query_texts=[query_text], n_results=top_k)
    matches = []
    if not results["ids"] or not results["ids"][0]:
        return matches

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]  # cosine distance, lower = closer
        if distance <= MATCH_THRESHOLD:
            meta = results["metadatas"][0][i]
            matches.append({
                "medicine_name": meta["medicine_name"],
                "batch_number": meta["batch_number"],
                "aliases": json.loads(meta["aliases"]),
                "manufacturer": meta["manufacturer"],
                "recall_status": meta.get("recall_status", "Recalled"),
                "is_recalled": meta.get("is_recalled", True),
                "recall_date": meta["recall_date"],
                "recall_reason": meta["recall_reason"],
                "recalling_agency": meta["recalling_agency"],
                "recall_class": meta["recall_class"],
                "recommendation": meta["recommendation"],
                "mfg_date": meta.get("mfg_date", ""),
                "expiry_date": meta.get("expiry_date", ""),
                "reporting_source": meta.get("reporting_source", ""),
                "reporting_lab": meta.get("reporting_lab", ""),
                "record_id": meta.get("record_id", ""),
                "distance": distance,
            })
    return matches
