"""modelkin: lineage-aware uncertainty for item-level LLM evaluation.

Models on a leaderboard are not independent respondents: fine-tunes and merges
of one base model answer alike. This package assigns models to base-model
families from repository metadata and reports item-level uncertainty with
families, not models, as the sampling unit.
"""
from .stats import ability, item_rest, residuals, residual_similarity
from .inference import (panel_card, effective_clusters, variance_inflation, family_bootstrap,
                        screen_items, similarity_by_family, leave_family_out)
from .lineage import assign_families, config_family, name_family, read_labels, write_labels

__version__ = "0.1.0"
__all__ = ["ability", "item_rest", "residuals", "residual_similarity", "panel_card", "effective_clusters",
           "variance_inflation", "family_bootstrap", "screen_items", "similarity_by_family", "leave_family_out",
           "assign_families", "config_family", "name_family", "read_labels", "write_labels"]
