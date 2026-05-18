from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from typing import Optional

import numpy as np


_INITIAL_TEXT: Optional[str] = None
_INITIAL_SIM: Optional[float] = None
_PREV_PHI: Optional[float] = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _hash_embedding(text: str, dims: int = 384) -> np.ndarray:
    vec = np.zeros(dims, dtype=float)
    for token in _tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


@lru_cache(maxsize=1)
def _load_st_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    model_name = os.environ.get("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _embed(text: str) -> np.ndarray:
    model = _load_st_model()
    if model is None:
        return _hash_embedding(text)

    emb = np.asarray(model.encode(text, normalize_embeddings=True), dtype=float)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _progress_bonus(text: str) -> float:
    lowered = text.lower()
    bonus = 0.0
    if "scan" in lowered or "discover" in lowered:
        bonus += 0.05
    if "exploit" in lowered or "user access" in lowered:
        bonus += 0.10
    if "privilege escalation" in lowered:
        bonus += 0.18
    if "root" in lowered:
        bonus += 0.22
    return bonus


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def set_initial_state(text: str) -> None:
    global _INITIAL_TEXT, _INITIAL_SIM, _PREV_PHI
    _INITIAL_TEXT = text
    _INITIAL_SIM = None
    _PREV_PHI = 0.0


def semantic_reward_with_baseline(text: str, goal_text: str) -> dict[str, float]:
    """
    Compute a completion-oriented semantic reward.

    Sentence-Transformers is used when available. In a fresh environment without
    the package or model weights, a deterministic hashing encoder keeps the
    benchmark runnable for smoke tests and plotting.
    """
    global _INITIAL_SIM, _PREV_PHI

    goal_emb = _embed(goal_text)
    cur_sim = _cosine(_embed(text), goal_emb)

    if _INITIAL_TEXT is None:
        set_initial_state(text)

    if _INITIAL_SIM is None:
        _INITIAL_SIM = _cosine(_embed(_INITIAL_TEXT or text), goal_emb)

    denom = max(1e-8, 1.0 - _INITIAL_SIM)
    semantic_phi = (cur_sim - _INITIAL_SIM) / denom
    phi = float(np.clip(semantic_phi + _progress_bonus(text), -1.0, 1.0))

    prev_phi = 0.0 if _PREV_PHI is None else _PREV_PHI
    delta_phi = phi - prev_phi
    _PREV_PHI = phi

    sigmoid_val = _sigmoid(6.0 * phi)
    reward = sigmoid_val

    return {
        "reward": float(reward),
        "potential": float(phi),
        "sigmoid_val": float(sigmoid_val),
        "delta_phi": float(delta_phi),
        "similarity": float(cur_sim),
        "baseline_similarity": float(_INITIAL_SIM),
    }
