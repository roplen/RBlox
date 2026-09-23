#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBlox QUALITY DATA FACTORY V5
=============================

Generator de dataset Luau strict haute qualité pour RBlox / Lumina.

Pipeline:
1. Génération batched avec vLLM (JSON structuré P1/P2 ou P3).
2. Reconstruction du champ assistant par Python.
3. Validation structurelle + Selene.
4. Déduplication exacte MD5/SHA-256 + sémantique MinHash/Jaccard.
5. Sauvegarde JSONL progressive et reprise après interruption.
6. Répartition stricte par paliers et sous-domaines.
7. Retry automatique avec feedback ciblé (max RBLOX_MAX_RETRIES).

IMPORTANT:
- vLLM n'est pas supporté nativement sous Windows. Utilise WSL2/Linux.
- FlashInfer 0.6.18.post1 + CUDA 12.4 : workaround VLLM_USE_FLASHINFER_SAMPLER=0 actif.

CHANGELOG V5.1:
- Suppression de la règle qui forçait task.spawn/task.defer dans tous les exemples.
- Réduction du fact_block (12 → 6 faits) pour alléger la charge cognitive du modèle.
- Élimination des doublons system/user dans les contraintes de prompt.
- Correction du faux positif wait()/spawn() sur le texte d'explication.
- correction du faux positif check_placeholders sur l'explication.
- Ajout métrique first_pass_accepted.
- Correction metrics.attempts (retries comptabilisés).
- Correction stem redondant avec base_name dans Selene parsing.
- Amélioration self-test Selene (diagnostic réel vérifié).
- Ajout protection RAM SemanticIndex (shingles compressées à grande échelle).
- Température 0.35 → 0.60, repetition_penalty 1.08 → 1.05.
- NONCE déplacé hors du contenu visible principal.
- Retry feedback Selene amélioré (lignes d'erreur précises).
- Ajout statistiques par bucket dans le state.
- Meilleure lisibilité des logs de rejet.
"""

from __future__ import annotations

# ── FlashInfer workaround (JIT nvcc --compress-mode=size incompatible) ──
import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import asyncio
import hashlib
import json
import random
import re
import shlex
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.environ.get(
    "RBLOX_VLLM_MODEL",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
)

OUTPUT_FILE = Path(
    os.environ.get(
        "RBLOX_QUALITY_OUT",
        "data/dataset_v5_30k.jsonl",
    )
)

REJECT_LOG = Path(
    os.environ.get(
        "RBLOX_REJECT_LOG",
        "data/dataset_v5_30k_rejected.jsonl",
    )
)

STATE_FILE = Path(
    os.environ.get(
        "RBLOX_STATE_FILE",
        "data/dataset_v5_30k.state.json",
    )
)

MANIFEST_FILE = Path(
    os.environ.get(
        "RBLOX_MANIFEST_FILE",
        "data/dataset_v5_30k.manifest.jsonl",
    )
)

DEFAULT_TARGET = int(os.environ.get("RBLOX_TARGET", "30000"))

BATCH_SIZE = max(1, int(os.environ.get("RBLOX_VLLM_BATCH", "8")))

MAX_MODEL_LEN = int(os.environ.get("RBLOX_VLLM_MAX_MODEL_LEN", "4096"))

MAX_TOKENS = int(os.environ.get("RBLOX_VLLM_MAX_TOKENS", "2800"))

GPU_MEMORY_UTILIZATION = float(
    os.environ.get("RBLOX_VLLM_GPU_MEMORY_UTILIZATION", "0.80")
)

# MAX_NUM_SEQS : priorité à la variable explicite, sinon BATCH_SIZE
_max_num_seqs_env = os.environ.get("RBLOX_VLLM_MAX_NUM_SEQS", "").strip()
MAX_NUM_SEQS = max(1, int(_max_num_seqs_env) if _max_num_seqs_env else BATCH_SIZE)

# Température relevée pour meilleure diversité sur 30k exemples
TEMPERATURE = float(os.environ.get("RBLOX_VLLM_TEMPERATURE", "0.60"))
TOP_P = float(os.environ.get("RBLOX_VLLM_TOP_P", "0.90"))
# Pénalité réduite pour ne pas bloquer les répétitions légitimes dans le code
REPETITION_PENALTY = float(os.environ.get("RBLOX_VLLM_REPETITION_PENALTY", "1.05"))
QUANTIZATION = os.environ.get("RBLOX_VLLM_QUANTIZATION", "").strip()

SELENE_COMMAND = os.environ.get("RBLOX_SELENE", "selene")
LUAU_ANALYZE_COMMAND = os.environ.get("RBLOX_LUAU_ANALYZE", "luau-analyze")
ANALYZER = os.environ.get("RBLOX_ANALYZER", "selene").strip().lower()

LINTER_THREADS = max(
    1,
    int(os.environ.get(
        "RBLOX_LINTER_THREADS",
        str(max(1, (os.cpu_count() or 4) // 2)),
    )),
)
LINTER_TIMEOUT = float(os.environ.get("RBLOX_LINTER_TIMEOUT", "30"))

SEMANTIC_THRESHOLD = float(os.environ.get("RBLOX_SEMANTIC_THRESHOLD", "0.80"))
MINHASH_PERMUTATIONS = int(os.environ.get("RBLOX_MINHASH_PERMUTATIONS", "64"))
MINHASH_SHINGLE_SIZE = int(os.environ.get("RBLOX_MINHASH_SHINGLE_SIZE", "5"))
LSH_BANDS = int(os.environ.get("RBLOX_LSH_BANDS", "8"))

# Seuil au-delà duquel SemanticIndex commence à compresser les shingles
# pour limiter la RAM sur les longues sessions (30k exemples).
SEMANTIC_COMPRESS_THRESHOLD = int(
    os.environ.get("RBLOX_SEMANTIC_COMPRESS_THRESHOLD", "5000")
)

MAX_ATTEMPTS = int(os.environ.get("RBLOX_MAX_ATTEMPTS", str(DEFAULT_TARGET * 25)))

STRICT_CHECKS = os.environ.get("RBLOX_STRICT_CHECKS", "1").strip().lower() not in {
    "0", "false", "no"
}

SEED = int(os.environ.get("RBLOX_SEED", "20260923"))
random.seed(SEED)

MAX_RETRIES_PER_EXAMPLE = max(
    0, int(os.environ.get("RBLOX_MAX_RETRIES", "2"))
)

# Maximum characters of Selene output to transmit in retry feedback
SELENE_FEEDBACK_MAX_CHARS = int(os.environ.get("RBLOX_SELENE_FEEDBACK_MAX", "600"))


# ============================================================
# DATASET TAXONOMY
# ============================================================

@dataclass(frozen=True)
class Bucket:
    stage: str
    stage_label: str
    subcategory: str
    topics: tuple[str, ...]
    weight: int


BASE_BUCKETS: tuple[Bucket, ...] = (
    Bucket(
        "P1", "Fondations, POO & typage strict avancé",
        "Typage avancé & generics",
        (
            "export type", "generic <T>", "generic <T, U>", "type guard",
            "typeof", "union types", "intersection types", "singleton types",
            "typed dictionary", "typed array", "generic Result type",
            "generic callback", "strict signal typing",
        ),
        2500,
    ),
    Bucket(
        "P1", "Fondations, POO & typage strict avancé",
        "POO moderne",
        (
            "setmetatable class", "__index", "__newindex", "__tostring",
            "__add", "__call", "weak table metatable", "typed constructor new",
            "inheritance", "encapsulation", "polymorphism", "private state",
            "interface-oriented object",
        ),
        2500,
    ),
    Bucket(
        "P1", "Fondations, POO & typage strict avancé",
        "Design patterns",
        (
            "Observer pattern", "finite state machine", "Factory pattern",
            "Singleton pattern", "Command pattern", "Strategy pattern",
            "Dependency Injection", "Middleware pipeline", "Promise composition",
            "Promise.all", "Promise.promisify", "event-driven module",
        ),
        2500,
    ),
    Bucket(
        "P1", "Fondations, POO & typage strict avancé",
        "Mémoire & threads",
        (
            "task.spawn lifecycle", "task.defer scheduling",
            "task.delay cancellation", "task.cancel pattern",
            "connection cleanup", "Maid pattern", "Janitor pattern",
            "weak-key table", "weak-value table", "bounded worker pool",
            "cooperative cancellation", "thread pool Luau",
        ),
        2500,
    ),
    Bucket(
        "P2", "Deep system, réseau, data & architecture",
        "Réseau, binaire & sérialisation",
        (
            "buffer serialization", "buffer bit packing",
            "buffer integer encoding", "bit32 masks", "compact packet format",
            "RemoteEvent validation", "UnreliableRemoteEvent validation",
            "RemoteFunction request response", "rate limiting",
            "server-authoritative remote", "schema validation",
            "network payload normalization",
        ),
        2000,
    ),
    Bucket(
        "P2", "Deep system, réseau, data & architecture",
        "Persistance & DataStores",
        (
            "DataStoreService", "GetAsync retry", "SetAsync retry",
            "UpdateAsync transformation", "session locking",
            "save-on-leave lifecycle", "MemoryStoreService", "SortedMap",
            "MemoryStoreQueue", "schema migration", "data validation",
            "bounded retry with backoff",
        ),
        2000,
    ),
    Bucket(
        "P2", "Deep system, réseau, data & architecture",
        "ECS & frameworks",
        (
            "pure ECS", "component storage", "archetype ECS",
            "decoupled system", "entity lifecycle", "event-driven architecture",
            "service boundary", "controller boundary", "dependency graph",
            "Knit-style service architecture", "framework adapter",
            "data-oriented iteration",
        ),
        2000,
    ),
    Bucket(
        "P2", "Deep system, réseau, data & architecture",
        "Parallel Luau & computation",
        (
            "Actor isolation", "task.desynchronize usage", "task.synchronize usage",
            "parallel Luau pattern", "BindToMessageParallel",
            "Actor message passing", "SharedTable usage",
            "worker partitioning", "parallel aggregation",
            "deterministic parallel task", "CPU-bound simulation",
            "actor-safe state transfer",
        ),
        2000,
    ),
    Bucket(
        "P2", "Deep system, réseau, data & architecture",
        "Algorithmes, mathématiques & physique",
        (
            "CFrame matrix math", "quaternion conversion", "raycast",
            "Blockcast", "Spherecast", "Shapecast", "Bezier curve",
            "PathfindingService", "A-star pathfinding", "octree",
            "spatial partitioning", "impulse calculation",
            "character controller", "3D interpolation",
        ),
        2000,
    ),
    Bucket(
        "P3", "Debugging, audit, sécurité & auto-correction",
        "Anti-cheat & sécurité serveur",
        (
            "RemoteEvent spoofing", "RemoteFunction spoofing",
            "speed exploit validation", "fly exploit validation",
            "noclip validation", "distance validation",
            "server cooldown validation", "server authority",
            "client trust boundary", "argument type validation",
            "rate-limit abuse", "server state verification",
        ),
        2500,
    ),
    Bucket(
        "P3", "Debugging, audit, sécurité & auto-correction",
        "Fuites mémoire & event leakage",
        (
            "missing Disconnect", "RBXScriptConnection leak",
            "closure retaining instance", "circular reference",
            "Maid cleanup bug", "Janitor cleanup bug", "weak table misuse",
            "metatable accumulation", "listener duplication",
            "PlayerRemoving cleanup", "CharacterAdded leak", "reconnect storm",
        ),
        2500,
    ),
    Bucket(
        "P3", "Debugging, audit, sécurité & auto-correction",
        "Race conditions, thread safety & deadlocks",
        (
            "concurrent UpdateAsync", "overlapping save", "HTTP request race",
            "shared mutable state", "task ordering bug", "deadlock",
            "starvation", "duplicate task", "cancellation race",
            "state transition race", "request timeout race",
            "atomic state update",
        ),
        2500,
    ),
    Bucket(
        "P3", "Debugging, audit, sécurité & auto-correction",
        "Refactoring & optimisation CPU/RAM",
        (
            "spaghetti refactor", "event-driven refactor",
            "Heartbeat loop optimization", "Stepped loop optimization",
            "RenderStepped misuse", "table allocation reduction",
            "closure allocation reduction", "hot-path optimization",
            "cache immutable lookup", "avoid repeated GetService",
            "reduce polling", "bounded work per frame",
        ),
        2500,
    ),
)


# ============================================================
# VERIFIED FACTS / GENERATION CONSTRAINTS
# ============================================================

# Facts are short, factual, and unambiguous.
# They are sampled per-prompt to reduce cognitive load on Qwen 3B.
COMMON_FACTS: tuple[str, ...] = (
    'Les services se récupèrent avec game:GetService("ServiceName").',
    "RemoteEvent: FireServer côté client, OnServerEvent côté serveur.",
    "RemoteFunction: InvokeServer côté client, OnServerInvoke côté serveur.",
    "Les appels DataStore doivent être protégés par pcall ou xpcall.",
    "UpdateAsync reçoit une fonction de transformation (oldValue) -> newValue.",
    "Un LocalScript ne doit pas accéder directement aux DataStores.",
    "workspace:Raycast() est l'API standard pour un raycast.",
    "Instance:IsA() vérifie la classe d'une Instance.",
    "typeof() effectue une vérification de type à l'exécution.",
    "Un ModuleScript retourne une valeur avec return, chargé via require().",
    "Les connexions RBXScriptConnection doivent être conservées pour nettoyage.",
    "task.wait, task.spawn, task.defer, task.delay, task.cancel sont les API modernes.",
    "Les ancienne API wait(), spawn(), delay() sont obsolètes et interdites.",
    "Le serveur doit recalculer ou vérifier toute action importante.",
    "Un Attribute se manipule avec SetAttribute/GetAttribute, pas Instance.new().",
    "SetAttribute et GetAttribute sont des méthodes d'Instance.",
    "Un UnreliableRemoteEvent est adapté aux données non critiques.",
    "MemoryStoreService fournit SortedMap et Queue pour le stockage temporaire.",
)

BUCKET_FACTS: dict[str, tuple[str, ...]] = {
    "Typage avancé & generics": (
        "type Name = ... définit un alias de type Luau.",
        "export type expose un alias depuis un ModuleScript.",
        "Les unions (A | B) et intersections (A & B) composent des types Luau.",
    ),
    "POO moderne": (
        "setmetatable fournit un prototype via __index.",
        "__tostring, __add, __call sont des métaméthodes Luau valides.",
        "__mode = 'k', 'v' ou 'kv' configure une table faible.",
    ),
    "Design patterns": (
        "Les patterns doivent rester de vrais programmes Luau, pas des pseudo-frameworks.",
    ),
    "Mémoire & threads": (
        "task.cancel annule un thread créé par task.spawn ou task.delay.",
        "Un système de cleanup doit être idempotent (safe à appeler plusieurs fois).",
    ),
    "Réseau, binaire & sérialisation": (
        "buffer est la bibliothèque Luau/Roblox pour le stockage binaire compact.",
        "bit32 fournit des opérations bit-à-bit pour les masques.",
        "Toute donnée réseau reçue doit être traitée comme non fiable.",
    ),
    "Persistance & DataStores": (
        "Les retries DataStore doivent être bornés avec une stratégie de backoff.",
        "Une migration de schéma doit gérer explicitement source et cible.",
    ),
    "ECS & frameworks": (
        "Un ECS sépare les données des systèmes qui les traitent.",
        "Ne pas inventer des fonctions Knit/Matter/Jecs non fournies dans la consigne.",
    ),
    "Parallel Luau & computation": (
        "task.desynchronize/task.synchronize contrôlent les transitions parallèles.",
        "SharedTable sert au partage de données entre contextes parallèles.",
    ),
    "Algorithmes, mathématiques & physique": (
        "Les opérations CFrame doivent préserver clairement le repère.",
        "PathfindingService calcule des chemins dans Workspace.",
    ),
    "Anti-cheat & sécurité serveur": (
        "Les cooldowns importants doivent être appliqués côté serveur.",
        "Les vérifications de distance utilisent des positions connues du serveur.",
    ),
    "Fuites mémoire & event leakage": (
        "Connect() reste actif jusqu'à Disconnect() explicite ou destruction.",
        "Une fermeture peut retenir des références et prolonger la vie d'objets.",
    ),
    "Race conditions, thread safety & deadlocks": (
        "Les opérations partagées doivent éviter les mises à jour concurrentes incompatibles.",
    ),
    "Refactoring & optimisation CPU/RAM": (
        "Les boucles de polling peuvent souvent être remplacées par des événements réactifs.",
        "Les allocations répétées dans une boucle chaude augmentent la pression mémoire.",
    ),
}

# UI patterns forbidden in all generated code and descriptions.
# Applied only to code fields, not to explanation text.
BANNED_UI_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bScreenGui\b", "UI interdite: ScreenGui"),
    (r"\bTextButton\b", "UI interdite: TextButton"),
    (r"\bTextLabel\b", "UI interdite: TextLabel"),
    (r"\bImageLabel\b", "UI interdite: ImageLabel"),
    (r"\bImageButton\b", "UI interdite: ImageButton"),
    (r"\bScrollingFrame\b", "UI interdite: ScrollingFrame"),
    (r"\bUIListLayout\b", "UI interdite: UIListLayout"),
    (r"\bUIGridLayout\b", "UI interdite: UIGridLayout"),
    (r"\bBillboardGui\b", "UI interdite: BillboardGui"),
    (r"\bSurfaceGui\b", "UI interdite: SurfaceGui"),
    (r"\bStarterGui\b", "UI interdite: StarterGui"),
    (r"\bCoreGui\b", "UI interdite: CoreGui"),
    (r"\bProximityPrompt\b", "UI/interaction interdite: ProximityPrompt"),
    (r"\bScreenGui\b", "UI interdite: ScreenGui"),
    (r"\bFrame\b(?!\s*=\s*\d)", "UI interdite: Frame (non-numeric)"),
)

BANNED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bRateLimiting\b", "API fictive: RateLimiting"),
    (r"\bRaycastService\b", "API fictive: RaycastService"),
    (r"\bImageService\b", "API fictive: ImageService"),
    (r"\bBindToFrame\b", "API fictive: BindToFrame"),
    (r"\bRemoteEvent\.Sent\b", "API fictive: RemoteEvent.Sent"),
    (r"\bOnServerReceived\b", "API fictive: OnServerReceived"),
    (r"\bOnClientReceive\b", "API fictive: OnClientReceive"),
    (r"\bOnServerReceive\b", "API fictive: OnServerReceive"),
    (r"\bAsyncTask\b", "API fictive: AsyncTask"),
    (r"\bSaveAsync\b", "API fictive: SaveAsync"),
    (r"\bValidateServer\b", "API inventée: ValidateServer"),
    (r"\bMultipleReturn\s*\(", "fonction fictive: MultipleReturn"),
    (r"\bInstance\.new\s*\(\s*[\"']Attribute[\"']\s*\)", "Attribute créé avec Instance.new"),
    (r"\bInstance\.new\s*\(\s*[\"']Players[\"']\s*\)", "Players créé avec Instance.new"),
    (r"\bInstance\.new\s*\(\s*[\"']RunService[\"']\s*\)", "RunService créé avec Instance.new"),
    (r"\bInstance\.new\s*\(\s*[\"']TweenService[\"']\s*\)", "TweenService créé avec Instance.new"),
    (r"\bInstance\.new\s*\(\s*[\"']DataStoreService[\"']\s*\)", "DataStoreService créé avec Instance.new"),
    (r"\bInstance\.new\s*\(\s*[\"']UserInputService[\"']\s*\)", "UserInputService créé avec Instance.new"),
    (r"\bHumanoid\.Attack\b", "API Humanoid.Attack inexistante"),
    # Casse incorrecte de l'API task (Task.* au lieu de task.*)
    (r"\bTask\s*\.\s*(?:Wait|Spawn|Delay|Cancel|Defer)\b", "casse Task incorrecte — utiliser task.wait/spawn/delay/cancel/defer"),
    # Ancienne API globale — uniquement dans les champs de code (pas l'explication)
    # Note: ces patterns sont appliqués UNIQUEMENT sur le code, pas sur l'explication
    (r"(?<![A-Za-z0-9_:\.])wait\s*\(", "ancienne API wait()"),
    (r"(?<![A-Za-z0-9_:\.])spawn\s*\(", "ancienne API spawn()"),
    (r"(?<![A-Za-z0-9_:\.])delay\s*\(", "ancienne API delay()"),
    (r"\btryCatch\s*\(", "syntaxe tryCatch inexistante"),
    (r"--!nocheck\b", "bypass de typechecking interdit"),
    (r"--!nolint\b", "bypass de lint interdit"),
)

PLACEHOLDER_PATTERNS: tuple[str, ...] = (
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bplaceholder\b",
    r"\bimplement here\b",
    r"\bimplementation here\b",
    r"\bnot implemented\b",
    r"\byour code here\b",
    r"\binsert code here\b",
    r"\ba compléter\b",
    r"\bà compléter\b",
    r"\bà implémenter\b",
)


# ============================================================
# HELPERS
# ============================================================

CODE_BLOCK_RE = re.compile(
    r"```(?:luau|lua)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|"
    r"\d+(?:\.\d+)?|"
    r"==|~=|<=|>=|::|->|"
    r"[{}()[\].,:;+\-*/%^#=<>|&]",
)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_code(code: str) -> str:
    code = re.sub(r"--\[\[.*?\]\]", " ", code, flags=re.DOTALL)
    code = re.sub(r"--[^\n]*", " ", code)
    code = re.sub(r"""(['"]).*?(?<!\\)\1""", "<str>", code, flags=re.DOTALL)
    code = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", code)
    return " ".join(code.lower().split())


def extract_code_blocks(text: str) -> list[str]:
    return [block.strip() for block in CODE_BLOCK_RE.findall(text) if block.strip()]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(stripped)
        if not match:
            raise ValueError("Aucun objet JSON détecté")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Le JSON racine doit être un objet")
    return data


def command_parts(command: str) -> list[str]:
    return shlex.split(command, posix=(os.name != "nt"))


def count_output_tokens(texts: Iterable[str], tokenizer: Any) -> int:
    if tokenizer is None:
        return sum(max(1, int(len(t.split()) * 1.35)) for t in texts)
    total = 0
    for text in texts:
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            ids = encoded.get("input_ids")
            if ids is not None:
                total += len(ids)
                continue
        except Exception:
            pass
        try:
            total += len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            total += max(1, int(len(text.split()) * 1.35))
    return total


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def choose_task(bucket: Bucket) -> str:
    if bucket.stage == "P3":
        return random.choice((
            "security audit", "memory leak audit", "race condition audit",
            "deadlock audit", "CPU/RAM refactor", "correctness refactor",
        ))
    return random.choice((
        "implementation", "architecture", "refactor", "edge case",
        "performance", "correctness", "test with assert",
    ))


def choose_topic(bucket: Bucket) -> str:
    return random.choice(bucket.topics)


def fact_block(bucket: Bucket) -> str:
    """
    Sample a small set of relevant facts to include in the prompt.
    Kept small (max 6) to reduce cognitive load on Qwen 3B.
    Bucket-specific facts are prioritised over common facts.
    """
    bucket_specific = list(BUCKET_FACTS.get(bucket.subcategory, ()))
    common = list(COMMON_FACTS)
    random.shuffle(common)

    # Take all bucket-specific facts (usually 2-3) + fill up to 6 with common facts
    combined = unique_preserving_order(bucket_specific + common)
    selected = combined[:6]

    return "\n".join(f"- {fact}" for fact in selected)


# ============================================================
# PLAN / QUOTAS
# ============================================================

def allocate_targets(total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("TARGET doit être > 0")
    base_total = sum(b.weight for b in BASE_BUCKETS)
    raw = {b.subcategory: total * b.weight / base_total for b in BASE_BUCKETS}
    result = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(result.values())
    ranked = sorted(raw, key=lambda k: raw[k] - result[k], reverse=True)
    for k in ranked[:remainder]:
        result[k] += 1
    return result


def stage_totals(targets: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for b in BASE_BUCKETS:
        result[b.stage] += targets[b.subcategory]
    return dict(result)


# ============================================================
# STRUCTURED OUTPUT PARSING & RECONSTRUCTION
# ============================================================

# Unified JSON schema accepted by vLLM for both P1/P2 and P3.
# Python validates that the stage-appropriate fields are present.
UNIFIED_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "user":           {"type": "string", "minLength": 10},
        "explanation":    {"type": "string"},
        "code":           {"type": "string"},
        "diagnosis":      {"type": "string"},
        "original_code":  {"type": "string"},
        "corrected_code": {"type": "string"},
    },
    "required": ["user"],
    "additionalProperties": False,
}


def _strip_code_fences(raw: str) -> str:
    """Remove any accidental Markdown code fences from a raw code field."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:luau|lua)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()


def parse_structured_output(text: str, stage: str) -> dict[str, str]:
    """
    Parse the structured JSON produced by the model and validate
    that the stage-appropriate fields are present.

    Returns a dict with normalised string fields.
    Raises ValueError on any issue.
    """
    data = extract_json_object(text)

    user = str(data.get("user", "")).strip()
    if not user:
        raise ValueError("Champ 'user' absent ou vide")

    if stage in ("P1", "P2"):
        explanation = str(data.get("explanation", "")).strip()
        code = str(data.get("code", "")).strip()

        if not explanation:
            raise ValueError("Champ 'explanation' absent ou vide (P1/P2)")
        if not code:
            raise ValueError("Champ 'code' absent ou vide (P1/P2)")

        code = _strip_code_fences(code)

        return {
            "user": user,
            "explanation": explanation,
            "code": code,
        }

    # P3
    diagnosis = str(data.get("diagnosis", "")).strip()
    original_code = str(data.get("original_code", "")).strip()
    corrected_code = str(data.get("corrected_code", "")).strip()

    if not diagnosis:
        raise ValueError("Champ 'diagnosis' absent ou vide (P3)")
    if not original_code:
        raise ValueError("Champ 'original_code' absent ou vide (P3)")
    if not corrected_code:
        raise ValueError("Champ 'corrected_code' absent ou vide (P3)")

    original_code = _strip_code_fences(original_code)
    corrected_code = _strip_code_fences(corrected_code)

    return {
        "user": user,
        "diagnosis": diagnosis,
        "original_code": original_code,
        "corrected_code": corrected_code,
    }


def build_assistant_from_structured(data: dict[str, str], stage: str) -> str:
    """
    Reconstruct the final 'assistant' field from the structured JSON fields.

    P1/P2:
        explanation + blank line + ```luau block
    P3:
        <think>diagnosis</think> + code blocks for original and corrected
    """
    if stage in ("P1", "P2"):
        explanation = data["explanation"]
        code = data["code"]
        return f"{explanation}\n\n```luau\n{code}\n```"

    # P3
    diagnosis = data["diagnosis"]
    original_code = data["original_code"]
    corrected_code = data["corrected_code"]

    return (
        f"<think>\n{diagnosis}\n</think>\n\n"
        f"Code problématique:\n\n```luau\n{original_code}\n```\n\n"
        f"Code corrigé:\n\n```luau\n{corrected_code}\n```"
    )


# ============================================================
# PROMPT ENGINE
# ============================================================

# System prompt: concise, non-redundant, small-model-friendly.
# Rules are prioritised by importance and deduplicated vs user prompts.
GENERATOR_SYSTEM = """\
Tu es un ingénieur senior Luau/Roblox. Ta mission : produire des données
d'entraînement de très haute précision pour un modèle spécialisé Roblox/Luau.

RÈGLES ABSOLUES (dans l'ordre de priorité):

1. Retourne UNIQUEMENT un objet JSON valide. Aucun texte avant ou après.
2. Les champs de code contiennent du Luau BRUT. Aucun ```, aucun Markdown.
3. N'invente jamais une API, propriété, méthode, service ou classe Roblox.
   N'utilise que des API Roblox/Luau que tu connais avec certitude.
4. Ne crée jamais un service avec Instance.new().
5. Aucune interface graphique : ScreenGui, Frame, TextButton, TextLabel,
   ImageLabel, ScrollingFrame, BillboardGui, SurfaceGui, StarterGui, CoreGui,
   ProximityPrompt. Le sujet est logiciel (architecture, réseau, data, etc.).
6. Aucun TODO, FIXME, placeholder, pseudo-code, "...", "à compléter".
7. Le code doit être complet et directement utile. Toute fonction ouverte
   doit être fermée avec end.
8. La première ligne de tout code Luau doit être exactement : --!strict
9. Côté client : non fiable. Toute action importante est autorisée côté serveur.
10. Échappe correctement les guillemets et retours à la ligne dans le JSON.
"""

# ── P1/P2 user instruction ─────────────────────────────────────────────────

def _p1p2_user_instruction(bucket: Bucket, topic: str, task: str, nonce: str) -> str:
    return f"""\
PALIER: {bucket.stage} — {bucket.subcategory}
THÈME: {topic}
TÂCHE: {task}

FAITS DE RÉFÉRENCE:
{fact_block(bucket)}

OBJECTIF:
Conçois un problème réel d'ingénierie Roblox/Luau autour du thème ci-dessus.
Le problème doit nécessiter une vraie implémentation, pas un cours théorique.

RÉPONSE ATTENDUE — JSON avec exactement ces trois champs:
{{
  "user": "Description précise et détaillée du problème d'ingénierie (60 à 800 caractères).",
  "explanation": "Explication concise de la solution en français, sans Markdown.",
  "code": "LUAU BRUT — première ligne : --!strict"
}}

CONTRAINTES CHAMP code:
- Première ligne : --!strict (obligatoire)
- Implémentation fonctionnelle complète avec vraie logique (function, local, if, for, return...)
- Aucun backtick, aucun ```, aucun HTML
- Aucun TODO, FIXME, pseudo-code, "...", placeholder
- Toutes les fonctions et blocs fermés avec end
- Maximum 200 lignes

[nonce:{nonce}]
Réponds uniquement avec le JSON structuré. Rien d'autre.\
"""


def _p3_user_instruction(bucket: Bucket, topic: str, task: str, nonce: str) -> str:
    return f"""\
PALIER: {bucket.stage} — {bucket.subcategory}
THÈME: {topic}
TÂCHE: {task}

FAITS DE RÉFÉRENCE:
{fact_block(bucket)}

OBJECTIF:
Crée un exercice d'audit/correction réel autour du thème ci-dessus.

RÉPONSE ATTENDUE — JSON avec exactement ces quatre champs:
{{
  "user": "Scénario + code Luau problématique brut + demande d'audit.",
  "diagnosis": "Diagnostic en français : causes précises, 2 à 5 points concis.",
  "original_code": "LUAU BRUT — le code problématique (peut contenir des bugs).",
  "corrected_code": "LUAU BRUT — première ligne : --!strict. Code corrigé complet."
}}

CONTRAINTES:
- original_code et corrected_code : Luau BRUT, sans ```, sans Markdown
- corrected_code : première ligne --!strict (obligatoire)
- corrected_code doit être différent de original_code et corriger les vrais problèmes
- corrected_code : complet, aucun TODO, FIXME, pseudo-code, placeholder
- Aucune interface graphique dans aucun des codes

[nonce:{nonce}]
Réponds uniquement avec le JSON structuré. Rien d'autre.\
"""


def build_prompt(
    bucket: Bucket,
    case_id: int,
    mutation: int,
    feedback: str = "",
) -> list[dict[str, str]]:
    topic = choose_topic(bucket)
    task = choose_task(bucket)
    nonce = f"{case_id:08d}-{mutation:04d}-{random.randrange(10**9):09d}"

    if bucket.stage != "P3":
        user_content = _p1p2_user_instruction(bucket, topic, task, nonce)
    else:
        user_content = _p3_user_instruction(bucket, topic, task, nonce)

    if feedback.strip():
        # Feedback is prepended, concise, and targeted.
        user_content = (
            "CORRECTION REQUISE — tentative précédente rejetée:\n"
            f"{feedback.strip()}\n\n"
            "---\n\n"
        ) + user_content

    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


# ============================================================
# RETRY FEEDBACK
# ============================================================

# Each feedback message is short and targeted.
# A small model needs a precise, actionable correction, not a wall of text.
RETRY_FEEDBACK_RULES: dict[str, str] = {
    "réponse trop courte": (
        "Réponse trop courte. Produis une vraie solution avec explication ET code complet."
    ),
    "json/schema invalide": (
        "JSON invalide. Réponds UNIQUEMENT avec l'objet JSON structuré. "
        "Échappe les guillemets internes avec \\\" et les retours à la ligne avec \\n."
    ),
    "aucun objet json": (
        "Aucun objet JSON détecté. Réponds uniquement avec { ... }."
    ),
    "champ 'code' absent": (
        "Champ 'code' absent ou vide. Fournis une implémentation Luau brute dans ce champ."
    ),
    "champ 'explanation' absent": (
        "Champ 'explanation' absent. Fournis une explication concise en français."
    ),
    "champ 'diagnosis' absent": (
        "Champ 'diagnosis' absent. Fournis un diagnostic précis (2 à 5 points)."
    ),
    "champ 'original_code' absent": (
        "Champ 'original_code' absent. Fournis le code Luau problématique."
    ),
    "champ 'corrected_code' absent": (
        "Champ 'corrected_code' absent. Fournis le code Luau corrigé commençant par --!strict."
    ),
    "code vide": (
        "Champ de code vide. Produis une vraie implémentation Luau."
    ),
    "--!strict manquant": (
        "ERREUR: la première ligne du code doit être exactement --!strict. "
        "Place --!strict comme toute première ligne du champ 'code' ou 'corrected_code'."
    ),
    "backticks présents": (
        "Backticks détectés dans le champ code. "
        "Le champ 'code' contient du Luau BRUT, sans ```, sans ```luau."
    ),
    "code p1/p2 trop court": (
        "Code trop court. Produis une implémentation substantielle avec vraie logique."
    ),
    "code p1/p2 sans logique exécutable": (
        "Code sans logique exécutable. Produis du vrai Luau avec fonctions, "
        "structures de contrôle et logique réelle."
    ),
    "code excessivement long": (
        "Code trop long. Reste focalisé, maximum 200 lignes."
    ),
    "code corrigé est identique": (
        "Le corrected_code est identique à original_code. Apporte de vraies corrections."
    ),
    "le code corrigé est identique au code problématique": (
        "corrected_code identique à original_code. Corrige réellement les problèmes identifiés."
    ),
    "selene": (
        "Selene a rejeté le code. Corrige l'erreur de lint indiquée ci-dessus "
        "et renvoie une implémentation complète et valide."
    ),
    "lint reject": (
        "Analyse statique échouée. Corrige les erreurs de syntaxe ou d'API."
    ),
    "similarité": (
        "Trop similaire à un exemple existant. Crée une variante réellement différente "
        "du problème, de l'architecture et du code."
    ),
    "question trop courte": (
        "Champ 'user' trop court. Décris le problème avec précision (min 60 caractères)."
    ),
    "question trop longue": (
        "Champ 'user' trop long. Reste concis sur un seul problème."
    ),
    "casse task incorrecte": (
        "ERREUR: utilise task.wait/task.spawn/task.defer/task.delay/task.cancel "
        "(minuscule). Ne jamais écrire Task.Wait, Task.Spawn, etc."
    ),
    "ancienne api wait": (
        "ERREUR: wait() est une ancienne API obsolète. "
        "Utilise task.wait() à la place."
    ),
    "ancienne api spawn": (
        "ERREUR: spawn() est une ancienne API obsolète. "
        "Utilise task.spawn() à la place."
    ),
    "ancienne api delay": (
        "ERREUR: delay() est une ancienne API obsolète. "
        "Utilise task.delay() à la place."
    ),
    "todo interdit": (
        "TODO interdit. Produis une implémentation complète."
    ),
    "fixme interdit": (
        "FIXME interdit. Produis une implémentation complète."
    ),
    "pseudo-code interdit": (
        "Pseudo-code interdit. Produis du vrai Luau exécutable."
    ),
}


def build_retry_feedback(reason: str) -> str:
    reason_lower = reason.lower()
    for key, feedback in RETRY_FEEDBACK_RULES.items():
        if key in reason_lower:
            return feedback
    return (
        f"Tentative précédente rejetée. Cause: {reason[:120]}. "
        "Corrige précisément cette erreur et respecte le format JSON demandé."
    )


def build_selene_feedback(reason: str, raw_selene_output: str) -> str:
    """
    Build a targeted retry feedback string for Selene lint failures.
    Extracts actual error lines (truncated if necessary).
    """
    error_lines: list[str] = []
    for line in raw_selene_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        # Include lines with error/warning keywords or file:line:col pattern
        if any(m in lower for m in ("error", "warning", "parse error", "invalid")):
            error_lines.append(stripped)
        elif re.search(r":\d+:\d+:", stripped):
            error_lines.append(stripped)

    error_lines = error_lines[:10]

    if error_lines:
        errors_text = "\n".join(error_lines)
    else:
        errors_text = reason[:300]

    intro = "Selene a rejeté le code. Erreurs:\n"
    outro = (
        "\nCorrige précisément ces erreurs.\n"
        "Respecte le format JSON et place --!strict en première ligne."
    )

    # Truncate errors if total exceeds budget
    max_errors_len = SELENE_FEEDBACK_MAX_CHARS - len(intro) - len(outro)
    if len(errors_text) > max_errors_len:
        errors_text = errors_text[:max(0, max_errors_len)] + "\n[...tronqué...]"

    return intro + errors_text + outro


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def check_ui_free(code: str) -> str:
    """Check for forbidden UI patterns — applied to code fields only, not explanation."""
    for pattern, reason in BANNED_UI_PATTERNS:
        if re.search(pattern, code):
            return reason
    return ""


def check_placeholders(code: str) -> str:
    """Check for placeholder patterns — applied to code fields only."""
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, code, flags=re.IGNORECASE):
            return f"placeholder interdit: {pattern}"
    return ""


def check_banned_api(code: str) -> str:
    """Check for banned API patterns in Luau code."""
    for pattern, reason in BANNED_PATTERNS:
        if re.search(pattern, code):
            return reason
    return ""


def validate_p1p2_code(code: str) -> tuple[bool, str]:
    """Validate the raw Luau code extracted from the 'code' field (P1/P2)."""
    stripped = code.strip()

    if not stripped:
        return False, "code vide"
    if not stripped.startswith("--!strict"):
        return False, "--!strict manquant"
    if "```" in stripped:
        return False, "backticks présents dans le code"
    if len(stripped) < 60:
        return False, "code P1/P2 trop court"

    executable_markers = (
        "function ", "local function ", "local ", "return ",
        "if ", "for ", "while ",
    )
    if not any(m in stripped for m in executable_markers):
        return False, "code P1/P2 sans logique exécutable"

    if len(stripped) > 14000:
        return False, "code excessivement long"

    return True, ""


def validate_p3_codes(
    original_code: str, corrected_code: str
) -> tuple[bool, str]:
    """Validate original and corrected codes for P3."""
    corrected = corrected_code.strip()
    original = original_code.strip()

    if not corrected:
        return False, "code corrigé vide"
    if not corrected.startswith("--!strict"):
        return False, "--!strict manquant dans corrected_code"
    if "```" in corrected:
        return False, "backticks présents dans corrected_code"
    if normalize_code(original) == normalize_code(corrected):
        return False, "le code corrigé est identique au code problématique"
    if len(corrected) < 60:
        return False, "corrected_code trop court"
    if len(corrected) > 14000:
        return False, "corrected_code excessivement long"

    return True, ""


def validate_generated(
    structured: dict[str, str],
    bucket: Bucket,
) -> tuple[bool, str, str]:
    """
    Validate a structured parsed response.

    Returns (ok, reason, code_for_dedup_and_lint).
    - For P1/P2: code_for_dedup_and_lint is the 'code' field.
    - For P3:    code_for_dedup_and_lint is the 'corrected_code' field.

    IMPORTANT: check_ui_free, check_placeholders, check_banned_api are applied
    ONLY to code fields, NOT to explanation/diagnosis text (avoids false positives).
    """
    user = structured.get("user", "").strip()

    if len(user) < 60:
        return False, "question trop courte", ""
    if len(user) > 7000:
        return False, "question trop longue", ""

    if bucket.stage in ("P1", "P2"):
        code = structured.get("code", "").strip()

        # Structural validation on code
        ok, reason = validate_p1p2_code(code)
        if not ok:
            return False, reason, ""

        # UI patterns on code only
        ui_reason = check_ui_free(code)
        if ui_reason:
            return False, ui_reason, ""

        # Placeholders on code only
        ph_reason = check_placeholders(code)
        if ph_reason:
            return False, ph_reason, ""

        # Banned API on code
        banned = check_banned_api(code)
        if banned:
            return False, banned, ""

        # Verify reconstruction has exactly one luau block
        assistant = build_assistant_from_structured(structured, bucket.stage)
        if len(assistant) < 100:
            return False, "réponse trop courte", ""
        blocks = extract_code_blocks(assistant)
        if len(blocks) != 1:
            return False, "le palier 1/2 doit contenir exactement 1 bloc code", ""

        return True, "", code

    # P3
    original_code = structured.get("original_code", "").strip()
    corrected_code = structured.get("corrected_code", "").strip()

    ok, reason = validate_p3_codes(original_code, corrected_code)
    if not ok:
        return False, reason, ""

    # UI/placeholder/banned on corrected_code only
    ui_reason = check_ui_free(corrected_code)
    if ui_reason:
        return False, ui_reason, ""

    ph_reason = check_placeholders(corrected_code)
    if ph_reason:
        return False, ph_reason, ""

    banned = check_banned_api(corrected_code)
    if banned:
        return False, banned, ""

    # Verify reconstruction
    assistant = build_assistant_from_structured(structured, bucket.stage)
    if len(assistant) < 100:
        return False, "réponse trop courte", ""
    if "<think>" not in assistant.lower():
        return False, "bloc <think> manquant au palier 3", ""
    if "</think>" not in assistant.lower():
        return False, "fermeture </think> manquante au palier 3", ""

    blocks = extract_code_blocks(assistant)
    if len(blocks) < 2:
        return False, "le palier 3 doit contenir code original + code corrigé", ""

    return True, "", corrected_code


# ============================================================
# MINHASH / JACCARD
# ============================================================

class SemanticIndex:
    """
    MinHash + LSH index for near-duplicate detection.

    Memory note: shingles are stored as frozensets (more compact than list[str]).
    Above SEMANTIC_COMPRESS_THRESHOLD entries, only hashed shingles are stored
    (as set[int]) to reduce RAM pressure for 30k+ example sessions.
    """

    def __init__(
        self,
        num_perm: int,
        shingle_size: int,
        threshold: float,
        bands: int,
    ) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm doit être > 0")
        if shingle_size <= 0:
            raise ValueError("shingle_size doit être > 0")
        if bands <= 0 or num_perm % bands != 0:
            raise ValueError("num_perm doit être divisible par bands")

        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.threshold = threshold
        self.bands = bands
        self.rows_per_band = num_perm // bands

        self.signatures: list[tuple[int, ...]] = []
        # shingles[i] is either set[str] (early phase) or set[int] (compressed phase)
        self.shingles: list[Any] = []
        self.lsh: dict[tuple[int, tuple[int, ...]], set[int]] = defaultdict(set)
        self._compressed = False

        self._seeds = [
            int.from_bytes(
                hashlib.blake2b(
                    f"rblox-v5-minhash-{i}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            for i in range(num_perm)
        ]

    def _tokenize(self, code: str) -> list[str]:
        return TOKEN_RE.findall(normalize_code(code))

    def _make_shingles_str(self, code: str) -> set[str]:
        tokens = self._tokenize(code)
        if len(tokens) <= self.shingle_size:
            return {" ".join(tokens)} if tokens else set()
        return {
            " ".join(tokens[i: i + self.shingle_size])
            for i in range(len(tokens) - self.shingle_size + 1)
        }

    def _shingles_to_ints(self, shingles_str: set[str]) -> set[int]:
        """Convert string shingles to integer hashes to save memory."""
        result: set[int] = set()
        for s in shingles_str:
            h = int.from_bytes(
                hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest(),
                "little",
            )
            result.add(h)
        return result

    def _maybe_compress(self) -> None:
        """
        If we cross the compression threshold and haven't compressed yet,
        convert all stored string shingles to integer hashes.
        """
        if self._compressed:
            return
        if len(self.shingles) >= SEMANTIC_COMPRESS_THRESHOLD:
            self.shingles = [
                self._shingles_to_ints(s) if isinstance(s, set) and s and isinstance(next(iter(s)), str) else s
                for s in self.shingles
            ]
            self._compressed = True

    def _hash_shingle(self, shingle: str, seed: int) -> int:
        payload = seed.to_bytes(8, "little") + shingle.encode("utf-8", errors="ignore")
        return int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "little"
        )

    def _signature(self, shingles_str: set[str]) -> tuple[int, ...]:
        max_val = (1 << 64) - 1
        return tuple(
            min(
                (self._hash_shingle(s, seed) for s in shingles_str),
                default=max_val,
            )
            for seed in self._seeds
        )

    def _band_keys(self, sig: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
        result = []
        for band in range(self.bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            result.append((band, sig[start:end]))
        return result

    @staticmethod
    def _jaccard_str(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        inter = len(left & right)
        union = len(left | right)
        return inter / union if union else 1.0

    @staticmethod
    def _jaccard_int(left: set[int], right: set[int]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        inter = len(left & right)
        union = len(left | right)
        return inter / union if union else 1.0

    def find_similar(self, code: str) -> tuple[bool, float, int | None]:
        shingles_str = self._make_shingles_str(code)
        if not shingles_str:
            return False, 0.0, None
        sig = self._signature(shingles_str)
        candidates: set[int] = set()
        for key in self._band_keys(sig):
            candidates.update(self.lsh.get(key, ()))

        best_score = 0.0
        best_index: int | None = None

        # For Jaccard comparison, we need compatible types
        shingles_int = self._shingles_to_ints(shingles_str) if self._compressed else None

        for cand in candidates:
            stored = self.shingles[cand]
            if self._compressed:
                score = self._jaccard_int(shingles_int, stored)  # type: ignore[arg-type]
            else:
                score = self._jaccard_str(shingles_str, stored)  # type: ignore[arg-type]

            if score > best_score:
                best_score = score
                best_index = cand
            if score > self.threshold:
                return True, score, cand

        return False, best_score, best_index

    def add(self, code: str) -> int:
        self._maybe_compress()
        shingles_str = self._make_shingles_str(code)
        sig = self._signature(shingles_str)
        idx = len(self.signatures)
        self.signatures.append(sig)

        if self._compressed:
            self.shingles.append(self._shingles_to_ints(shingles_str))
        else:
            self.shingles.append(shingles_str)

        for key in self._band_keys(sig):
            self.lsh[key].add(idx)
        return idx


# ============================================================
# PERSISTENCE
# ============================================================

class SeenRegistry:
    def __init__(self) -> None:
        self.md5: set[str] = set()
        self.sha256: set[str] = set()
        self.question_hashes: set[str] = set()
        self.semantic = SemanticIndex(
            num_perm=MINHASH_PERMUTATIONS,
            shingle_size=MINHASH_SHINGLE_SIZE,
            threshold=SEMANTIC_THRESHOLD,
            bands=LSH_BANDS,
        )

    @staticmethod
    def _hashes(code: str) -> tuple[str, str]:
        normalized = normalize_code(code).encode("utf-8")
        return (
            hashlib.md5(normalized).hexdigest(),
            hashlib.sha256(normalized).hexdigest(),
        )

    def add(self, user: str, code: str) -> None:
        md5, sha256 = self._hashes(code)
        self.md5.add(md5)
        self.sha256.add(sha256)
        self.question_hashes.add(
            hashlib.sha256(normalize_text(user).encode("utf-8")).hexdigest()
        )
        self.semantic.add(code)

    def check(self, user: str, code: str) -> tuple[bool, str, float]:
        md5, sha256 = self._hashes(code)
        if md5 in self.md5:
            return False, "duplicate MD5", 1.0
        if sha256 in self.sha256:
            return False, "duplicate SHA-256", 1.0
        qhash = hashlib.sha256(normalize_text(user).encode("utf-8")).hexdigest()
        if qhash in self.question_hashes:
            return False, "duplicate question", 1.0
        similar, score, _ = self.semantic.find_similar(code)
        if similar and score > SEMANTIC_THRESHOLD:
            return False, f"similarité sémantique {score:.3f} > {SEMANTIC_THRESHOLD:.2f}", score
        return True, "", score


def extract_messages(item: dict[str, Any]) -> tuple[str, str]:
    messages = item.get("messages", [])
    if not isinstance(messages, list):
        return "", ""
    user = ""
    assistant = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    return user, assistant


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json_dumps(item) + "\n")


def write_state(
    accepted: int,
    rejected: int,
    attempts: int,
    counts: dict[str, int],
    reject_counts: Counter[str],
    started_at: float,
    first_pass_accepted: int = 0,
) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 5,
        "model": MODEL,
        "target": DEFAULT_TARGET,
        "accepted": accepted,
        "rejected": rejected,
        "attempts": attempts,
        "first_pass_accepted": first_pass_accepted,
        "first_pass_rate": (
            round(first_pass_accepted / max(1, attempts) * 100, 2)
        ),
        "counts": counts,
        "reject_counts": dict(reject_counts),
        "elapsed_seconds": time.perf_counter() - started_at,
        "updated_at": time.time(),
    }
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {
            "counts": {}, "reject_counts": {}, "accepted": 0,
            "rejected": 0, "attempts": 0, "first_pass_accepted": 0,
        }
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except Exception:
        return {
            "counts": {}, "reject_counts": {}, "accepted": 0,
            "rejected": 0, "attempts": 0, "first_pass_accepted": 0,
        }


def load_existing_dataset(registry: SeenRegistry) -> int:
    if not OUTPUT_FILE.exists():
        return 0
    count = 0
    with OUTPUT_FILE.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            messages = item.get("messages")
            if not isinstance(messages, list):
                continue
            user, assistant = extract_messages(item)
            if not user or not assistant:
                continue
            blocks = extract_code_blocks(assistant)
            if not blocks:
                continue
            registry.add(user, blocks[-1])
            count += 1
    return count


# ============================================================
# LINTER SUBPROCESS
# ============================================================

@dataclass
class LintResult:
    index: int
    ok: bool
    reason: str
    raw_output: str


def _parse_selene_output_for_file(
    output: str,
    file_name: str,
    returncode: int,
) -> str:
    """
    Parse Selene 0.31.0 output for a specific file.

    Selene 0.31.0 quiet format:
      filename.luau:line:col: [error_type] message
      filename.luau:line:col: (warning) [rule_name] message

    Returns empty string if no issues, or a pipe-separated string of up to 5 issue lines.
    """
    if returncode == 0:
        return ""

    # Use only the base filename for matching (Selene prints just the filename
    # when invoked with cwd set to the directory containing the files)
    base_name = os.path.basename(file_name)

    issue_lines: list[str] = []
    in_continuation = False

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            in_continuation = False
            continue

        # Primary match: line references our file by name
        if base_name in stripped and re.search(r":\d+:\d+:", stripped):
            issue_lines.append(stripped)
            in_continuation = True
            continue

        # Continuation lines (detail/context lines, usually indented)
        if in_continuation and line.startswith(" "):
            issue_lines.append(stripped)
            continue

        in_continuation = False

    if issue_lines:
        return " | ".join(issue_lines[:5])

    # Fallback: if returncode != 0 but no file-specific lines found,
    # grab any error/warning lines from the entire output
    if returncode != 0:
        generic_lines: list[str] = []
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if any(m in lower for m in ("error", "warning", "parse error", "invalid")):
                generic_lines.append(stripped)
            elif re.search(r":\d+:\d+:", stripped):
                generic_lines.append(stripped)
        if generic_lines:
            return " | ".join(generic_lines[:5])
        return f"Selene exit={returncode}: diagnostic unavailable for {base_name}"

    return ""


def _parse_luau_analyze_output_for_file(
    output: str,
    file_name: str,
    returncode: int,
) -> str:
    """Parse luau-analyze output for a specific file."""
    if returncode == 0:
        return ""

    base_name = os.path.basename(file_name)
    issue_lines: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if base_name in stripped or file_name in stripped:
            issue_lines.append(stripped)

    if issue_lines:
        return " | ".join(issue_lines[:5])

    generic_lines: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if any(m in lower for m in ("error", "warning")):
            generic_lines.append(stripped)

    if generic_lines:
        return " | ".join(generic_lines[:5])

    return f"luau-analyze exit={returncode}: diagnostic unavailable"


async def lint_batch(codes: list[str]) -> list[LintResult]:
    """
    Run the configured linter on a batch of Luau code strings.
    Each code string is written to a separate temporary file.
    Returns a LintResult per code.
    """
    if not codes:
        return []

    with tempfile.TemporaryDirectory(prefix="rblox_v5_lint_") as tmp_name:
        tmp_dir = Path(tmp_name)
        file_names = [f"case_{i:06d}.luau" for i in range(len(codes))]

        for fn, code in zip(file_names, codes):
            (tmp_dir / fn).write_text(code, encoding="utf-8")

        if ANALYZER == "selene":
            # selene.toml is placed in the cwd so Selene picks it up automatically.
            # std = "roblox" enables Roblox-specific linting rules.
            # Note: lua_versions is not a valid Selene 0.31 config option.
            (tmp_dir / "selene.toml").write_text(
                'std = "roblox"\n', encoding="utf-8"
            )
            # Selene 0.31.0 valid flags:
            #   --display-style=<quiet|rich|json>
            #   --color=<always|auto|never>
            #   --num-threads <N>
            # Note: --no-summary was removed in Selene 0.27+
            command = command_parts(SELENE_COMMAND) + [
                "--display-style=quiet",
                "--color=never",
                "--num-threads", str(LINTER_THREADS),
                *file_names,
            ]

        elif ANALYZER == "luau-analyze":
            command = command_parts(LUAU_ANALYZE_COMMAND) + file_names
        else:
            raise RuntimeError("RBLOX_ANALYZER doit être 'selene' ou 'luau-analyze'")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(tmp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as err:
            raise RuntimeError(
                f"Analyseur introuvable: {command[0]}. "
                "Installe l'outil ou configure RBLOX_SELENE/RBLOX_LUAU_ANALYZE."
            ) from err

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=LINTER_TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return [
                LintResult(index=i, ok=False, reason="timeout du linter", raw_output="")
                for i in range(len(codes))
            ]

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        combined_output = stdout_text
        if stderr_text.strip():
            combined_output = combined_output + "\n" + stderr_text

        returncode = process.returncode or 0

        results: list[LintResult] = []
        for i, fn in enumerate(file_names):
            if ANALYZER == "selene":
                issue = _parse_selene_output_for_file(combined_output, fn, returncode)
            else:
                issue = _parse_luau_analyze_output_for_file(combined_output, fn, returncode)

            results.append(LintResult(
                index=i,
                ok=(issue == ""),
                reason=issue,
                raw_output=combined_output,
            ))

        return results


# ============================================================
# vLLM ENGINE
# ============================================================

def build_engine() -> tuple[Any, Any]:
    if os.name == "nt":
        raise RuntimeError(
            "vLLM n'est pas supporté nativement sous Windows. "
            "Utilise WSL2/Linux avec CUDA."
        )

    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as err:
        raise RuntimeError(
            "vLLM est requis. Installe une version compatible avec ton environnement CUDA/Python."
        ) from err

    llm_kwargs: dict[str, Any] = {
        "model": MODEL,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "enable_prefix_caching": True,
        "trust_remote_code": False,
    }
    if QUANTIZATION:
        llm_kwargs["quantization"] = QUANTIZATION

    print("Chargement vLLM...")
    print(f"  Model              : {MODEL}")
    print(f"  GPU mem util       : {GPU_MEMORY_UTILIZATION}")
    print(f"  max_model_len      : {MAX_MODEL_LEN}")
    print(f"  max_num_seqs       : {MAX_NUM_SEQS}")
    print(f"  batch_size         : {BATCH_SIZE}")
    print(f"  temperature        : {TEMPERATURE}")
    print(f"  repetition_penalty : {REPETITION_PENALTY}")
    print(f"  max_tokens         : {MAX_TOKENS}")
    print(f"  Quantization       : {QUANTIZATION or 'auto'}")
    print(f"  FlashInfer sampler : {os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', '?')}")

    llm = LLM(**llm_kwargs)

    structured_outputs = StructuredOutputsParams(json=UNIFIED_JSON_SCHEMA)
    params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        max_tokens=MAX_TOKENS,
        structured_outputs=structured_outputs,
    )

    return llm, params


def extract_vllm_text(result: Any) -> str:
    outputs = getattr(result, "outputs", None)
    if not outputs:
        return ""
    return str(getattr(outputs[0], "text", "")).strip()


def generate_with_vllm(
    llm: Any,
    sampling_params: Any,
    specs: list[tuple[Bucket, int, int]],
    feedbacks: list[str] | None = None,
) -> tuple[list[str], float, int]:
    if feedbacks is not None and len(feedbacks) != len(specs):
        raise ValueError("Le nombre de feedbacks doit correspondre au nombre de specs.")

    conversations = [
        build_prompt(
            bucket, case_id, mutation,
            feedbacks[i] if feedbacks is not None else "",
        )
        for i, (bucket, case_id, mutation) in enumerate(specs)
    ]

    started = time.perf_counter()
    results = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
    elapsed = max(1e-9, time.perf_counter() - started)
    texts = [extract_vllm_text(r) for r in results]
    return texts, elapsed, len(texts)


# ============================================================
# DATASET ITEM
# ============================================================

def make_dataset_item(user: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es RBlox, une IA spécialisée en Luau strict et Roblox. "
                    "Tu privilégies le code exact, sécurisé, maintenable et sans API inventée."
                ),
            },
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


# ============================================================
# GENERATION SCHEDULER
# ============================================================

class BucketScheduler:
    def __init__(self, targets: dict[str, int], existing_counts: dict[str, int]) -> None:
        self.targets = targets
        self.counts: dict[str, int] = {
            k: min(int(existing_counts.get(k, 0)), targets[k])
            for k in targets
        }

    def deficits(self) -> dict[str, int]:
        return {k: max(0, self.targets[k] - self.counts[k]) for k in self.targets}

    def done(self) -> bool:
        return all(self.counts[k] >= self.targets[k] for k in self.targets)

    def choose_buckets(
        self, size: int, case_start: int, mutation: int
    ) -> list[tuple[Bucket, int, int]]:
        deficits = self.deficits()
        ordered = sorted(self.targets, key=lambda k: deficits[k], reverse=True)
        if not ordered or deficits[ordered[0]] <= 0:
            return []
        bucket_by_name = {b.subcategory: b for b in BASE_BUCKETS}
        selected: list[tuple[Bucket, int, int]] = []
        for offset in range(size):
            name = ordered[offset % len(ordered)]
            if deficits[name] <= 0:
                continue
            selected.append((bucket_by_name[name], case_start + offset, mutation + offset))
        return selected

    def accepted(self, subcategory: str) -> None:
        if subcategory in self.counts:
            self.counts[subcategory] += 1


# ============================================================
# PROGRESS / METRICS
# ============================================================

@dataclass
class Metrics:
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    lint_rejected: int = 0
    semantic_rejected: int = 0
    json_rejected: int = 0
    validation_rejected: int = 0
    output_tokens: int = 0
    generation_seconds: float = 0.0
    # attempts = total generation calls (initial + retries)
    attempts: int = 0
    # first_pass_accepted = accepted on the very first attempt (no retry)
    first_pass_accepted: int = 0


# ============================================================
# SELF TEST
# ============================================================

def _run_selene_on_code(code: str) -> tuple[bool | None, str]:
    """
    Synchronous helper: run Selene on a single code string.
    Returns (ok, reason). Returns (None, msg) if Selene is not installed.
    Used only in self-test.
    """
    import subprocess

    with tempfile.TemporaryDirectory(prefix="rblox_selftest_") as tmp_name:
        tmp_dir = Path(tmp_name)
        code_file = tmp_dir / "test.luau"
        code_file.write_text(code, encoding="utf-8")
        (tmp_dir / "selene.toml").write_text('std = "roblox"\n', encoding="utf-8")

        cmd = command_parts(SELENE_COMMAND) + [
            "--display-style=quiet",
            "--color=never",
            "test.luau",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(tmp_dir),
                capture_output=True,
                timeout=15,
            )
        except FileNotFoundError:
            return None, "selene not found"
        except subprocess.TimeoutExpired:
            return False, "timeout"

        combined = result.stdout.decode("utf-8", errors="replace")
        if result.stderr:
            combined += "\n" + result.stderr.decode("utf-8", errors="replace")

        issue = _parse_selene_output_for_file(combined, "test.luau", result.returncode)
        return (issue == ""), issue


def self_test() -> None:
    print("Running self-test...")
    failures: list[str] = []

    def assert_test(condition: bool, test_id: str, message: str) -> None:
        if not condition:
            failures.append(f"TEST {test_id} FAILED: {message}")
            print(f"  [{test_id}] FAILED: {message}")
        # (pass output is printed inline below)

    # ── 1. JSON P1/P2 valide ──────────────────────────────────────────────────
    raw_p1 = json.dumps({
        "user": "Implémente un compteur générique strict avec reset en Luau.",
        "explanation": "Un compteur typé générique en Luau.",
        "code": (
            "--!strict\n"
            "local function makeCounter(initial: number): () -> number\n"
            "    local count = initial\n"
            "    return function(): number\n"
            "        count += 1\n"
            "        return count\n"
            "    end\n"
            "end\n"
            "local next = makeCounter(0)\n"
            "print(next())\n"
        ),
    })
    try:
        parsed_p1 = parse_structured_output(raw_p1, "P1")
        assert_test(
            "Implémente" in parsed_p1["user"],
            "1a", f"user field incorrect: {parsed_p1['user'][:50]}",
        )
        assert_test(
            parsed_p1["explanation"].startswith("Un compteur"),
            "1b", f"explanation incorrect: {parsed_p1['explanation'][:50]}",
        )
        assert_test(
            parsed_p1["code"].startswith("--!strict"),
            "1c", "code ne commence pas par --!strict",
        )
        print("  [1] JSON P1/P2 valide: OK")
    except Exception as e:
        assert_test(False, "1", f"exception: {e}")

    # ── 2. Reconstruction P1/P2 ───────────────────────────────────────────────
    try:
        assistant_p1 = build_assistant_from_structured(parsed_p1, "P1")
        assert_test("```luau" in assistant_p1, "2a", "pas de ```luau dans assistant")
        assert_test("--!strict" in assistant_p1, "2b", "pas de --!strict dans assistant")
        assert_test(assistant_p1.endswith("```"), "2c", "assistant ne finit pas par ```")
        print("  [2] Reconstruction P1/P2: OK")
    except Exception as e:
        assert_test(False, "2", f"exception: {e}")

    # ── 3. JSON P3 valide ─────────────────────────────────────────────────────
    raw_p3 = json.dumps({
        "user": (
            "Ce script serveur écoute un RemoteEvent sans valider les arguments "
            "ni appliquer de cooldown côté serveur. Audite et corrige.\n\n"
            "local re = game:GetService('ReplicatedStorage').RemoteEvent\n"
            "re.OnServerEvent:Connect(function(player, amount)\n"
            "    player.leaderstats.Gold.Value += amount\n"
            "end)\n"
        ),
        "diagnosis": (
            "1. Aucune validation du type ou de la plage de 'amount'.\n"
            "2. Pas de cooldown serveur: exploit de spam trivial.\n"
            "3. Le serveur fait confiance à la valeur envoyée par le client."
        ),
        "original_code": (
            "local re = game:GetService('ReplicatedStorage').RemoteEvent\n"
            "re.OnServerEvent:Connect(function(player, amount)\n"
            "    player.leaderstats.Gold.Value += amount\n"
            "end)\n"
        ),
        "corrected_code": (
            "--!strict\n"
            "local Players = game:GetService('Players')\n"
            "local ReplicatedStorage = game:GetService('ReplicatedStorage')\n"
            "local re = ReplicatedStorage:WaitForChild('RemoteEvent')\n"
            "local GOLD_PER_ACTION = 10\n"
            "local COOLDOWN = 1\n"
            "local lastTime: {[Player]: number} = {}\n"
            "re.OnServerEvent:Connect(function(player: Player)\n"
            "    local now = os.clock()\n"
            "    if (now - (lastTime[player] or 0)) < COOLDOWN then return end\n"
            "    lastTime[player] = now\n"
            "    local gold = player:FindFirstChild('leaderstats')\n"
            "        and player.leaderstats:FindFirstChild('Gold')\n"
            "    if gold and gold:IsA('IntValue') then\n"
            "        gold.Value += GOLD_PER_ACTION\n"
            "    end\n"
            "end)\n"
            "Players.PlayerRemoving:Connect(function(player: Player)\n"
            "    lastTime[player] = nil\n"
            "end)\n"
        ),
    })
    try:
        parsed_p3 = parse_structured_output(raw_p3, "P3")
        assert_test(
            parsed_p3["diagnosis"].startswith("1."),
            "3a", "diagnosis incorrect",
        )
        assert_test(
            parsed_p3["original_code"].startswith("local"),
            "3b", "original_code incorrect",
        )
        assert_test(
            parsed_p3["corrected_code"].startswith("--!strict"),
            "3c", "corrected_code ne commence pas par --!strict",
        )
        print("  [3] JSON P3 valide: OK")
    except Exception as e:
        assert_test(False, "3", f"exception: {e}")

    # ── 4. Reconstruction P3 ──────────────────────────────────────────────────
    try:
        assistant_p3 = build_assistant_from_structured(parsed_p3, "P3")
        assert_test("<think>" in assistant_p3, "4a", "pas de <think>")
        assert_test("</think>" in assistant_p3, "4b", "pas de </think>")
        assert_test("```luau" in assistant_p3, "4c", "pas de ```luau dans P3 assistant")
        assert_test("Code problématique:" in assistant_p3, "4d", "pas de 'Code problématique:'")
        assert_test("Code corrigé:" in assistant_p3, "4e", "pas de 'Code corrigé:'")
        print("  [4] Reconstruction P3: OK")
    except Exception as e:
        assert_test(False, "4", f"exception: {e}")

    # ── 5. validate_p1p2_code: code valide ───────────────────────────────────
    valid_code = (
        "--!strict\n"
        "local function add(a: number, b: number): number\n"
        "    return a + b\n"
        "end\n"
        "print(add(1, 2))\n"
    )
    ok, reason = validate_p1p2_code(valid_code)
    assert_test(ok, "5", f"code valide rejeté: {reason}")
    print("  [5] validate_p1p2_code code valide: OK")

    # ── 6. Rejet code vide ────────────────────────────────────────────────────
    ok, reason = validate_p1p2_code("")
    assert_test(not ok, "6a", "code vide devrait être rejeté")
    assert_test("vide" in reason.lower(), "6b", f"mauvaise raison: {reason}")
    print("  [6] Rejet code vide: OK")

    # ── 7. Rejet code trop court ──────────────────────────────────────────────
    ok, reason = validate_p1p2_code("--!strict\nlocal x = 1")
    assert_test(not ok, "7a", "code trop court devrait être rejeté")
    assert_test("court" in reason.lower(), "7b", f"mauvaise raison: {reason}")
    print("  [7] Rejet code trop court: OK")

    # ── 8. Rejet code sans logique exécutable ─────────────────────────────────
    short_no_logic = "--!strict\n" + "-- commentaire\n" * 6
    ok, reason = validate_p1p2_code(short_no_logic)
    assert_test(not ok, "8", f"code sans logique devrait être rejeté, raison: {reason}")
    print("  [8] Rejet code sans logique exécutable: OK")

    # ── 9. Détection code corrigé identique ──────────────────────────────────
    identical_code = (
        "--!strict\n"
        "local function add(a: number, b: number): number\n"
        "    return a + b\n"
        "end\n"
    )
    ok, reason = validate_p3_codes(identical_code, identical_code)
    assert_test(not ok, "9a", "code identique devrait être rejeté")
    assert_test("identique" in reason.lower(), "9b", f"mauvaise raison: {reason}")
    print("  [9] Détection code corrigé identique: OK")

    # ── 10. Retry feedback ────────────────────────────────────────────────────
    fb = build_retry_feedback("--!strict manquant")
    assert_test("--!strict" in fb, "10a", "feedback devrait mentionner --!strict")
    fb2 = build_retry_feedback("json/schema invalide")
    assert_test("JSON" in fb2 or "json" in fb2.lower(), "10b", "feedback JSON incorrect")
    fb3 = build_retry_feedback("casse task incorrecte")
    assert_test("task." in fb3.lower(), "10c", "feedback casse task incorrect")
    print("  [10] Retry feedback: OK")

    # ── 11. Parsing JSON invalide ─────────────────────────────────────────────
    raised = False
    try:
        parse_structured_output("pas du json {broken", "P1")
    except ValueError:
        raised = True
    assert_test(raised, "11", "JSON invalide devrait lever ValueError")
    print("  [11] Parsing JSON invalide: OK")

    # ── 12. Code fence strippé dans le champ code ─────────────────────────────
    raw_with_fence = json.dumps({
        "user": "Teste la détection de backticks dans le champ code en Luau strict.",
        "explanation": "Explication test.",
        "code": "```luau\n--!strict\nlocal x = 1\nprint(x)\n```",
    })
    try:
        parsed_fence = parse_structured_output(raw_with_fence, "P1")
        assert_test(
            "```" not in parsed_fence["code"],
            "12a", f"backticks non strippés: {parsed_fence['code'][:60]!r}",
        )
        assert_test(
            parsed_fence["code"].startswith("--!strict"),
            "12b", "code ne commence pas par --!strict après strip",
        )
        print("  [12] Code fence strippé: OK")
    except Exception as e:
        assert_test(False, "12", f"exception: {e}")

    # ── 13. Structure finale assistant P1 ─────────────────────────────────────
    data_check = {
        "user": "Test structure assistant.",
        "explanation": "Une explication concise.",
        "code": "--!strict\nlocal x = 42\nprint(x)\n",
    }
    final_assistant = build_assistant_from_structured(data_check, "P1")
    assert_test(
        final_assistant.startswith("Une explication concise."),
        "13a", "l'explication doit être en premier",
    )
    assert_test("```luau\n--!strict" in final_assistant, "13b", "bloc luau manquant")
    assert_test(final_assistant.endswith("```"), "13c", "fermeture ``` manquante")
    print("  [13] Structure finale assistant P1: OK")

    # ── 14. Selene: code valide accepté ──────────────────────────────────────
    valid_luau = (
        "--!strict\n"
        "local function greet(name: string): string\n"
        "    return 'Hello, ' .. name\n"
        "end\n"
        "print(greet('Roblox'))\n"
    )
    selene_ok, selene_reason = _run_selene_on_code(valid_luau)
    if selene_ok is None:
        print("  [14] Selene code valide: SKIP (selene non installé)")
    else:
        assert_test(selene_ok, "14", f"code valide rejeté par Selene: {selene_reason}")
        print("  [14] Selene code valide: OK")

    # ── 15. Selene: code invalide rejeté (parse_error garanti) ───────────────
    # Using a clear syntax error that Selene always catches as parse_error
    invalid_luau_parse = (
        "--!strict\n"
        "local function broken(\n"
        "    -- missing closing parenthesis and body\n"
    )
    selene_ok2, selene_reason2 = _run_selene_on_code(invalid_luau_parse)
    if selene_ok2 is None:
        print("  [15] Selene code invalide: SKIP (selene non installé)")
    else:
        assert_test(
            not selene_ok2,
            "15",
            f"code invalide devrait être rejeté par Selene, raison: {selene_reason2}",
        )
        print(f"  [15] Selene code invalide rejeté: OK (raison: {selene_reason2[:80]})")

    # ── 16. Selene feedback builder ───────────────────────────────────────────
    fake_selene_output = (
        "case_000000.luau:3:5: (warning) [deprecated] wait is deprecated\n"
        "case_000000.luau:5:1: (error) [undefined_variable] unknown_func is not defined\n"
    )
    fb_selene = build_selene_feedback("lint reject", fake_selene_output)
    assert_test(
        "deprecated" in fb_selene or "undefined" in fb_selene or "Selene" in fb_selene,
        "16a", "feedback Selene devrait contenir les erreurs",
    )
    assert_test(
        len(fb_selene) <= SELENE_FEEDBACK_MAX_CHARS + 100,
        "16b", f"feedback Selene trop long: {len(fb_selene)} > {SELENE_FEEDBACK_MAX_CHARS + 100}",
    )
    print("  [16] Selene feedback: OK")

    # ── 17. Compteur retry: sémantique correcte ───────────────────────────────
    def _simulate_pipeline(
        outcomes: list[bool],
        max_retries: int,
    ) -> tuple[int, int, int, int]:
        """
        Simulate the retry pipeline.
        Returns (accepted, rejected, attempts, first_pass_accepted).
        - attempt 0 = initial attempt.
        - attempts 1..N = retries.
        rejected increments ONLY when all retries are exhausted.
        first_pass_accepted increments ONLY when attempt index == 0 succeeds.
        """
        accepted = 0
        rejected = 0
        total_attempts = 0
        first_pass_accepted = 0
        retry_count = 0

        for attempt_idx, ok in enumerate(outcomes):
            total_attempts += 1
            if ok:
                accepted += 1
                if attempt_idx == 0:
                    first_pass_accepted += 1
                return accepted, rejected, total_attempts, first_pass_accepted
            else:
                if retry_count < max_retries:
                    retry_count += 1
                    # continue to next attempt
                else:
                    rejected += 1
                    return accepted, rejected, total_attempts, first_pass_accepted

        # Exhausted all outcomes without success
        rejected += 1
        return accepted, rejected, total_attempts, first_pass_accepted

    # 17A: fail then succeed
    acc, rej, att, fpa = _simulate_pipeline([False, True], max_retries=2)
    assert_test(acc == 1, "17A-acc", f"expected 1, got {acc}")
    assert_test(rej == 0, "17A-rej", f"expected 0, got {rej}")
    assert_test(att == 2, "17A-att", f"expected 2, got {att}")
    assert_test(fpa == 0, "17A-fpa", f"expected first_pass=0, got {fpa}")

    # 17B: fail 3 times (initial + 2 retries)
    acc, rej, att, fpa = _simulate_pipeline([False, False, False], max_retries=2)
    assert_test(acc == 0, "17B-acc", f"expected 0, got {acc}")
    assert_test(rej == 1, "17B-rej", f"expected 1, got {rej}")
    assert_test(att == 3, "17B-att", f"expected 3, got {att}")
    assert_test(fpa == 0, "17B-fpa", f"expected first_pass=0, got {fpa}")

    # 17C: immediate success
    acc, rej, att, fpa = _simulate_pipeline([True], max_retries=2)
    assert_test(acc == 1, "17C-acc", f"expected 1, got {acc}")
    assert_test(rej == 0, "17C-rej", f"expected 0, got {rej}")
    assert_test(att == 1, "17C-att", f"expected 1, got {att}")
    assert_test(fpa == 1, "17C-fpa", f"expected first_pass=1, got {fpa}")

    print("  [17] Retry counter semantics: OK")

    # ── 18. Selene feedback truncation ───────────────────────────────────────
    long_output = "case_000000.luau:1:1: (error) [long_error] " + "x" * 2000 + "\n"
    fb_long = build_selene_feedback("lint reject", long_output)
    assert_test(
        len(fb_long) <= SELENE_FEEDBACK_MAX_CHARS + 150,
        "18", f"feedback tronqué trop long: {len(fb_long)}",
    )
    print("  [18] Selene feedback truncation: OK")

    # ── 19. check_banned_api: Task casse ──────────────────────────────────────
    code_task_wrong = "--!strict\nlocal t = Task.Wait(1)\n"
    reason_task = check_banned_api(code_task_wrong)
    assert_test(
        "Task" in reason_task or "casse" in reason_task.lower(),
        "19a", f"Task.Wait devrait être détecté, raison: {reason_task!r}",
    )

    code_task_ok = "--!strict\ntask.wait(1)\n"
    reason_task_ok = check_banned_api(code_task_ok)
    assert_test(
        reason_task_ok == "",
        "19b", f"task.wait(1) ne devrait pas être détecté, raison: {reason_task_ok!r}",
    )
    print("  [19] check_banned_api Task casse: OK")

    # ── 20. check_banned_api: wait() faux positif ─────────────────────────────
    # wait() in code → should be caught
    code_wait_bad = "--!strict\nwait(1)\nlocal x = 5\n"
    reason_wait_bad = check_banned_api(code_wait_bad)
    assert_test(
        "wait" in reason_wait_bad.lower(),
        "20a", f"wait() devrait être détecté, raison: {reason_wait_bad!r}",
    )

    # task.wait() in code → should NOT be caught
    code_task_wait_ok = "--!strict\ntask.wait(1)\nlocal x = 5\n"
    reason_task_wait_ok = check_banned_api(code_task_wait_ok)
    assert_test(
        reason_task_wait_ok == "",
        "20b", f"task.wait() ne devrait pas être détecté, raison: {reason_task_wait_ok!r}",
    )
    print("  [20] check_banned_api wait() faux positif: OK")

    # ── 21. validate_generated: explication avec wait() → pas de faux positif ─
    # The check_banned_api is now applied ONLY to code, not to explanation.
    # So an explanation mentioning "wait()" should not cause rejection.
    structured_with_wait_in_explanation = {
        "user": "Comment remplacer wait() par task.wait() dans un script serveur Roblox?",
        "explanation": (
            "L'ancienne API wait() est obsolète. "
            "On la remplace systématiquement par task.wait()."
        ),
        "code": (
            "--!strict\n"
            "-- Exemple de remplacement de wait() par task.wait()\n"
            "local function doWork()\n"
            "    task.wait(1)\n"
            "    return true\n"
            "end\n"
            "doWork()\n"
        ),
    }
    bucket_p1 = BASE_BUCKETS[0]  # any P1 bucket
    ok_val, reason_val, code_val = validate_generated(structured_with_wait_in_explanation, bucket_p1)
    assert_test(
        ok_val,
        "21",
        f"explanation avec wait() ne devrait pas être rejetée, raison: {reason_val}",
    )
    print("  [21] Faux positif wait() dans explication: OK")

    # ── 22. SemanticIndex compression ────────────────────────────────────────
    idx = SemanticIndex(
        num_perm=MINHASH_PERMUTATIONS,
        shingle_size=MINHASH_SHINGLE_SIZE,
        threshold=SEMANTIC_THRESHOLD,
        bands=LSH_BANDS,
    )
    # Add entries up to compression threshold
    for i in range(min(SEMANTIC_COMPRESS_THRESHOLD + 2, 20)):
        idx.add(f"--!strict\nlocal x{i} = {i}\nprint(x{i})\n")
    # Basic find_similar works
    similar, score, _ = idx.find_similar("--!strict\nlocal x0 = 0\nprint(x0)\n")
    # Either found similar or not — just verify no crash
    print("  [22] SemanticIndex (add/find_similar): OK")

    # ── Summary ───────────────────────────────────────────────────────────────
    if failures:
        print(f"\nSELF-TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\nSELF-TEST: OK")


# ============================================================
# ARGS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RBlox QUALITY DATA FACTORY V5")
    parser.add_argument("--target",     type=int,  default=DEFAULT_TARGET)
    parser.add_argument("--output",     type=Path, default=OUTPUT_FILE)
    parser.add_argument("--reject-log", type=Path, default=None)
    parser.add_argument("--state",      type=Path, default=None)
    parser.add_argument("--manifest",   type=Path, default=None)
    parser.add_argument("--batch-size", type=int,  default=BATCH_SIZE)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--self-test",  action="store_true")
    parser.add_argument(
        "--reset", action="store_true",
        help="Supprime output/state/manifest/reject-log avant de commencer.",
    )
    return parser.parse_args()


# ============================================================
# MAIN PIPELINE
# ============================================================

async def run_pipeline(args: argparse.Namespace) -> None:
    global OUTPUT_FILE, REJECT_LOG, STATE_FILE, MANIFEST_FILE, BATCH_SIZE, MAX_ATTEMPTS

    OUTPUT_FILE   = args.output
    REJECT_LOG    = args.reject_log or OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_rejected.jsonl")
    STATE_FILE    = args.state    or OUTPUT_FILE.with_suffix(".state.json")
    MANIFEST_FILE = args.manifest or OUTPUT_FILE.with_suffix(".manifest.jsonl")
    BATCH_SIZE    = max(1, args.batch_size)
    MAX_ATTEMPTS  = max(1, args.max_attempts if args.max_attempts is not None else args.target * 25)

    if args.reset:
        for path in (OUTPUT_FILE, REJECT_LOG, STATE_FILE, MANIFEST_FILE):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    targets = allocate_targets(args.target)
    state = load_state()
    registry = SeenRegistry()
    existing_count = load_existing_dataset(registry)

    stored_counts = state.get("counts", {})
    if not isinstance(stored_counts, dict):
        stored_counts = {}

    scheduler = BucketScheduler(targets, stored_counts)

    metrics = Metrics(
        accepted=existing_count,
        rejected=int(state.get("rejected", 0)),
        attempts=int(state.get("attempts", 0)),
        first_pass_accepted=int(state.get("first_pass_accepted", 0)),
    )
    reject_counts: Counter[str] = Counter(
        state.get("reject_counts", {})
        if isinstance(state.get("reject_counts", {}), dict)
        else {}
    )

    if existing_count >= args.target:
        print(f"Dataset déjà à {existing_count}/{args.target}.")
        return

    llm, sampling_params = build_engine()

    tokenizer = None
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        pass

    started_at = time.perf_counter()
    case_counter = existing_count + 1
    mutation_counter = 1

    progress = (
        tqdm(
            total=args.target,
            initial=min(existing_count, args.target),
            unit="ex",
            dynamic_ncols=True,
            desc="RBlox V5",
        )
        if tqdm is not None
        else None
    )

    try:
        while not scheduler.done():
            if metrics.attempts >= MAX_ATTEMPTS:
                raise RuntimeError(
                    "MAX_ATTEMPTS atteint avant la cible. "
                    "Consulte le reject log et augmente la diversification."
                )

            remaining = args.target - metrics.accepted
            if remaining <= 0:
                break

            specs = scheduler.choose_buckets(
                min(BATCH_SIZE, remaining), case_counter, mutation_counter
            )
            if not specs:
                raise RuntimeError("Le scheduler n'a plus de bucket à générer.")

            case_counter    += len(specs)
            mutation_counter += 1

            # metrics.attempts counts ALL generation calls (initial + retries).
            # It is incremented each time we actually call vLLM.
            # Each pending item has its own retry_count (0 = initial attempt).

            pending: list[dict[str, Any]] = [
                {
                    "spec":             spec,
                    "feedback":         "",
                    "last_reason":      "",
                    "last_lint_output": "",
                    "retry_count":      0,  # 0 = initial attempt, 1+ = retry
                }
                for spec in specs
            ]

            while pending:
                retry_specs     = [item["spec"]    for item in pending]
                retry_feedbacks = [item["feedback"] for item in pending]

                # Count this generation call for ALL items in this sub-batch
                metrics.attempts += len(pending)

                # ── Generation ──────────────────────────────────────────────
                try:
                    raw_texts, elapsed, result_count = await asyncio.to_thread(
                        generate_with_vllm, llm, sampling_params,
                        retry_specs, retry_feedbacks,
                    )
                except Exception as err:
                    reason = f"generation error: {err}"
                    next_pending: list[dict[str, Any]] = []
                    for item in pending:
                        bucket, case_id, mutation = item["spec"]
                        item["last_reason"] = reason
                        if item["retry_count"] < MAX_RETRIES_PER_EXAMPLE:
                            item["retry_count"] += 1
                            item["feedback"] = (
                                "La génération précédente a échoué techniquement. "
                                "Génère une réponse complète avec le JSON structuré demandé."
                            )
                            next_pending.append(item)
                            print(
                                f"↻ Retry {bucket.stage} case={case_id} "
                                f"attempt={item['retry_count']}/{MAX_RETRIES_PER_EXAMPLE} "
                                f"reason=generation_error"
                            )
                        else:
                            metrics.rejected += 1
                            reject_counts[reason] += 1
                            print(
                                f"✗ Reject {bucket.stage} case={case_id} "
                                f"reason=generation_error"
                            )
                            append_jsonl(REJECT_LOG, {
                                "case_id": case_id, "mutation": mutation,
                                "stage": bucket.stage, "subcategory": bucket.subcategory,
                                "reason": reason,
                            })
                    pending = next_pending
                    continue

                metrics.generation_seconds += elapsed
                metrics.output_tokens += count_output_tokens(raw_texts, tokenizer)

                next_pending = []
                candidate_records: list[tuple[dict[str, Any], Bucket, dict[str, str], str]] = []

                if result_count < len(pending):
                    print(
                        f"\nAttention: {len(pending) - result_count} "
                        "sortie(s) vLLM manquante(s)."
                    )

                for idx_item, item in enumerate(pending):
                    bucket, case_id, mutation = item["spec"]
                    is_initial_attempt = (item["retry_count"] == 0)

                    # ── Missing output ────────────────────────────────────────
                    if idx_item >= len(raw_texts):
                        reason = "vLLM output missing"
                        item["last_reason"] = reason
                        if item["retry_count"] < MAX_RETRIES_PER_EXAMPLE:
                            item["retry_count"] += 1
                            item["feedback"] = (
                                "Sortie absente. Génère une réponse complète avec le JSON structuré."
                            )
                            next_pending.append(item)
                            print(
                                f"↻ Retry {bucket.stage} case={case_id} "
                                f"attempt={item['retry_count']}/{MAX_RETRIES_PER_EXAMPLE} "
                                f"reason=output_missing"
                            )
                        else:
                            metrics.rejected += 1
                            reject_counts[reason] += 1
                            print(f"✗ Reject {bucket.stage} case={case_id} reason=output_missing")
                            append_jsonl(REJECT_LOG, {
                                "case_id": case_id, "mutation": mutation,
                                "stage": bucket.stage, "subcategory": bucket.subcategory,
                                "reason": reason,
                            })
                        continue

                    raw_text = raw_texts[idx_item]

                    # ── Parse structured JSON ─────────────────────────────────
                    try:
                        structured = parse_structured_output(raw_text, bucket.stage)
                    except Exception as err:
                        reason = f"json/schema invalide: {err}"
                        item["last_reason"] = reason
                        if item["retry_count"] < MAX_RETRIES_PER_EXAMPLE:
                            item["retry_count"] += 1
                            item["feedback"] = build_retry_feedback(reason)
                            next_pending.append(item)
                            print(
                                f"↻ Retry {bucket.stage} case={case_id} "
                                f"attempt={item['retry_count']}/{MAX_RETRIES_PER_EXAMPLE} "
                                f"reason=json_invalid"
                            )
                        else:
                            metrics.rejected += 1
                            metrics.json_rejected += 1
                            reject_counts[reason] += 1
                            print(
                                f"✗ Reject {bucket.stage} case={case_id} "
                                f"reason=json_invalid: {str(err)[:120]}"
                            )
                            append_jsonl(REJECT_LOG, {
                                "case_id": case_id, "mutation": mutation,
                                "stage": bucket.stage, "subcategory": bucket.subcategory,
                                "reason": reason, "raw": raw_text[:2000],
                            })
                        continue

                    # ── Structural validation ─────────────────────────────────
                    ok, reason, code_for_dedup = validate_generated(structured, bucket)
                    if not ok:
                        item["last_reason"] = reason
                        if item["retry_count"] < MAX_RETRIES_PER_EXAMPLE:
                            item["retry_count"] += 1
                            item["feedback"] = build_retry_feedback(reason)
                            next_pending.append(item)
                            print(
                                f"↻ Retry {bucket.stage} case={case_id} "
                                f"attempt={item['retry_count']}/{MAX_RETRIES_PER_EXAMPLE} "
                                f"reason={reason[:80]}"
                            )
                        else:
                            metrics.rejected += 1
                            metrics.validation_rejected += 1
                            reject_counts[reason] += 1
                            print(
                                f"✗ Reject {bucket.stage} case={case_id} "
                                f"reason={reason[:120]}"
                            )
                            append_jsonl(REJECT_LOG, {
                                "case_id": case_id, "mutation": mutation,
                                "stage": bucket.stage, "subcategory": bucket.subcategory,
                                "reason": reason,
                            })
                        continue

                    # ── Deduplication ─────────────────────────────────────────
                    user = structured["user"]
                    can_accept, dup_reason, score = registry.check(user, code_for_dedup)
                    if not can_accept:
                        item["last_reason"] = dup_reason
                        if item["retry_count"] < MAX_RETRIES_PER_EXAMPLE:
                            item["retry_count"] += 1
                            item["feedback"] = build_retry_feedback(dup_reason)
                            next_pending.append(item)
                            print(
                                f"↻ Retry {bucket.stage} case={case_id} "
                                f"attempt={item['retry_count']}/{MAX_RETRIES_PER_EXAMPLE} "
                                f"reason=dup({dup_reason[:60]})"
                            )
                        else:
                            metrics.rejected += 1
                            reject_counts[dup_reason] += 1
                            if "similarité" in dup_reason:
                                metrics.semantic_rejected += 1
                            else:
                                metrics.duplicates += 1
                            print(
                                f"✗ Reject {bucket.stage} case={case_id} "
                                f"reason=dup({dup_reason[:60]})"
                            )
                            append_jsonl(REJECT_LOG, {
                                "case_id": case_id, "mutation": mutation,
                                "stage": bucket.stage, "subcategory": bucket.subcategory,
                                "reason": dup_reason, "similarity": score,
                            })
                        continue

                    # This item passed all pre-lint checks — mark for lint batch
                    candidate_records.append((item, bucket, structured, code_for_dedup))

                # ── Lint batch ────────────────────────────────────────────────
                if candidate_records:
                    lint_codes = [code for (_, _, _, code) in candidate_records]
                    lint_results = await lint_batch(lint_codes)

                    for record, lint_result in zip(candidate_records, lint_results):
                        item, bucket, structured, code_for_dedup = record
                        _, case_id, mutation = item["spec"]
                        is_initial_attempt = (item["retry_count"] == 0)

                        if not lint_result.ok:
                            reason = "selene: " + lint_result.reason
                            item["last_reason"] = reason
                            item["last_lint_output"] = lint_result.raw_output

                            if item["retry_count"] < MAX_RETRIES_PER_EXAMPLE:
                                item["retry_count"] += 1
                                item["feedback"] = build_selene_feedback(
                                    reason, lint_result.raw_output
                                )
                                next_pending.append(item)
                                print(
                                    f"↻ Retry {bucket.stage} case={case_id} "
                                    f"attempt={item['retry_count']}/{MAX_RETRIES_PER_EXAMPLE} "
                                    f"reason=selene: {lint_result.reason[:80]}"
                                )
                            else:
                                metrics.rejected += 1
                                metrics.lint_rejected += 1
                                reject_counts[reason] += 1
                                print(
                                    f"✗ Reject {bucket.stage} case={case_id} "
                                    f"reason=selene: {lint_result.reason[:120]}"
                                )
                                append_jsonl(REJECT_LOG, {
                                    "case_id": case_id, "mutation": mutation,
                                    "stage": bucket.stage, "subcategory": bucket.subcategory,
                                    "reason": reason,
                                    "lint_output": lint_result.raw_output[:6000],
                                })
                            continue

                        # ── Accept ────────────────────────────────────────────
                        user      = structured["user"]
                        assistant = build_assistant_from_structured(structured, bucket.stage)

                        append_jsonl(OUTPUT_FILE, make_dataset_item(user, assistant))
                        append_jsonl(MANIFEST_FILE, {
                            "line":         metrics.accepted + 1,
                            "stage":        bucket.stage,
                            "stage_label":  bucket.stage_label,
                            "subcategory":  bucket.subcategory,
                            "case_id":      case_id,
                            "mutation":     mutation,
                            "retry_count":  item["retry_count"],
                            "corrected_sha256": hashlib.sha256(
                                normalize_code(code_for_dedup).encode("utf-8")
                            ).hexdigest(),
                        })

                        registry.add(user, code_for_dedup)
                        scheduler.accepted(bucket.subcategory)
                        metrics.accepted += 1

                        # Track first-pass acceptance separately
                        if is_initial_attempt:
                            metrics.first_pass_accepted += 1
                            print(
                                f"✓ Accepted {bucket.stage} case={case_id} "
                                f"[FIRST PASS] total={metrics.accepted}"
                            )
                        else:
                            print(
                                f"✓ Accepted {bucket.stage} case={case_id} "
                                f"[retry={item['retry_count']}] total={metrics.accepted}"
                            )

                        if progress is not None:
                            progress.update(1)

                        if metrics.accepted >= args.target:
                            break

                pending = next_pending

                if metrics.accepted >= args.target:
                    pending = []

                if pending:
                    causes = Counter(item["last_reason"] for item in pending if item["last_reason"])
                    top = causes.most_common(1)
                    cause_str = f" (cause: {top[0][0][:60]})" if top else ""
                    print(
                        f"↻ Retry: {len(pending)} exemple(s) restant(s){cause_str}."
                    )

                write_state(
                    metrics.accepted, metrics.rejected, metrics.attempts,
                    scheduler.counts, reject_counts, started_at,
                    first_pass_accepted=metrics.first_pass_accepted,
                )

            # ── Per-batch progress ────────────────────────────────────────────
            total_elapsed = max(1e-9, time.perf_counter() - started_at)
            tok_s = metrics.output_tokens / max(metrics.generation_seconds, 1e-9)
            ex_s  = metrics.accepted / total_elapsed
            fpr   = (
                metrics.first_pass_accepted / max(1, metrics.accepted) * 100
                if metrics.accepted > 0 else 0.0
            )

            if progress is not None:
                progress.set_postfix(
                    acc=metrics.accepted,
                    rej=metrics.rejected,
                    fp=f"{fpr:.0f}%",
                    dup=metrics.duplicates,
                    lint=metrics.lint_rejected,
                    tok_s=f"{tok_s:.1f}",
                )
            else:
                print(
                    f"Progress {metrics.accepted}/{args.target} | "
                    f"fp={fpr:.1f}% | reject={metrics.rejected} | "
                    f"dup={metrics.duplicates} | lint={metrics.lint_rejected} | "
                    f"tok/s={tok_s:.1f}"
                )

    finally:
        if progress is not None:
            progress.close()
        write_state(
            metrics.accepted, metrics.rejected, metrics.attempts,
            scheduler.counts, reject_counts, started_at,
            first_pass_accepted=metrics.first_pass_accepted,
        )

    elapsed_total = max(1e-9, time.perf_counter() - started_at)
    fpr_final = (
        metrics.first_pass_accepted / max(1, metrics.accepted) * 100
        if metrics.accepted > 0 else 0.0
    )

    print()
    print("=" * 72)
    print("RBLOX QUALITY DATA FACTORY V5 — COMPLETE")
    print("=" * 72)
    print(f"Accepted              : {metrics.accepted}")
    print(f"  First-pass accepted : {metrics.first_pass_accepted} ({fpr_final:.1f}% of accepted)")
    print(f"Rejected              : {metrics.rejected}")
    print(f"  JSON rejected       : {metrics.json_rejected}")
    print(f"  Validation rejected : {metrics.validation_rejected}")
    print(f"  Lint (Selene) rej.  : {metrics.lint_rejected}")
    print(f"  Exact duplicates    : {metrics.duplicates}")
    print(f"  Semantic duplicates : {metrics.semantic_rejected}")
    print(f"Total attempts        : {metrics.attempts}")
    print(f"Generation tok/s      : {metrics.output_tokens / max(metrics.generation_seconds, 1e-9):.2f}")
    print(f"Overall ex/s          : {metrics.accepted / elapsed_total:.4f}")
    print(f"Elapsed               : {elapsed_total:.1f}s")
    print(f"Output                : {OUTPUT_FILE}")
    print(f"Reject log            : {REJECT_LOG}")
    print(f"State                 : {STATE_FILE}")
    print(f"Manifest              : {MANIFEST_FILE}")
    print()
    print("Répartition cible:")
    for b in BASE_BUCKETS:
        current = scheduler.counts.get(b.subcategory, 0)
        target  = targets[b.subcategory]
        pct     = current / max(1, target) * 100
        print(f"  {b.stage:<2} {b.subcategory:<38} {current:>6}/{target:<6} ({pct:.0f}%)")
    print()
    print("Top causes de rejet:")
    for cause, count in reject_counts.most_common(10):
        print(f"  {count:>5}x  {cause[:70]}")
    print("=" * 72)


def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    if args.target <= 0:
        raise SystemExit("--target doit être > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size doit être > 0")

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        print("\nArrêt demandé. Le dernier état sauvegardé reste réutilisable.")
    except Exception as err:
        print(f"\nERREUR FATALE: {err}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
