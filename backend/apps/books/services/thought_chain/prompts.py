from __future__ import annotations

from django.conf import settings

THOUGHT_CHAIN_MODE = "llm_thought_chain"
THOUGHT_SAME_BLOCK_THRESHOLD = float(getattr(settings, "THOUGHT_SAME_BLOCK_THRESHOLD", 0.65))
THOUGHT_RELATION_THRESHOLD = float(getattr(settings, "THOUGHT_RELATION_THRESHOLD", 0.65))
THOUGHT_BLOCK_MEMBERSHIP_THRESHOLD = float(getattr(settings, "THOUGHT_BLOCK_MEMBERSHIP_THRESHOLD", 0.70))
DEFAULT_DRY_RUN_MAX_PAIRS = int(getattr(settings, "THOUGHT_CHAIN_DRY_RUN_MAX_PAIRS", 60))
EXISTING_BLOCK_LIMIT = int(getattr(settings, "THOUGHT_CHAIN_EXISTING_BLOCK_LIMIT", 30))
