"""
Parameter registry and tuning framework for controlled research iteration.
"""

from tuning.campaigns import default_group_tuning_plans, run_group_tuning_campaign
from tuning.comparison import (
    build_candidate_vs_production_report,
    build_parameter_change_table,
)
from tuning.experiments import run_parameter_experiment
from tuning.governance import (
    evaluate_current_production_signal_quality,
    get_candidate_review_context,
    materialize_candidate_parameter_pack,
    run_controlled_tuning_workflow,
)
from tuning.packs import load_parameter_pack, list_parameter_packs
from tuning.promotion import (
    evaluate_promotion,
    get_active_live_pack,
    get_active_shadow_pack,
    get_manual_approval_record,
    get_promotion_runtime_context,
    move_candidate_to_shadow,
    promote_candidate,
    record_manual_approval,
    rollback_live_pack,
    update_pack_state,
)
from tuning.registry import get_parameter_registry
from tuning.reporting import summarize_promotion_workflow
from tuning.runtime import (
    get_active_parameter_pack,
    get_parameter_value,
    set_active_parameter_pack,
    temporary_parameter_pack,
)
from tuning.shadow import compare_shadow_trade_outputs, summarize_shadow_log
from tuning.search import (
    run_coordinate_descent_search,
    run_grid_search,
    run_latin_hypercube_search,
    run_random_search,
)
from tuning.validation import compare_validation_results, run_walk_forward_validation
from tuning.walk_forward import build_walk_forward_splits

__all__ = [
    "build_walk_forward_splits",
    "build_candidate_vs_production_report",
    "build_parameter_change_table",
    "default_group_tuning_plans",
    "compare_validation_results",
    "compare_shadow_trade_outputs",
    "evaluate_current_production_signal_quality",
    "evaluate_promotion",
    "get_candidate_review_context",
    "get_active_live_pack",
    "get_active_parameter_pack",
    "get_manual_approval_record",
    "get_parameter_registry",
    "get_parameter_value",
    "get_active_shadow_pack",
    "get_promotion_runtime_context",
    "list_parameter_packs",
    "load_parameter_pack",
    "move_candidate_to_shadow",
    "materialize_candidate_parameter_pack",
    "promote_candidate",
    "record_manual_approval",
    "rollback_live_pack",
    "run_coordinate_descent_search",
    "run_controlled_tuning_workflow",
    "run_grid_search",
    "run_group_tuning_campaign",
    "run_latin_hypercube_search",
    "run_parameter_experiment",
    "run_random_search",
    "run_walk_forward_validation",
    "set_active_parameter_pack",
    "summarize_promotion_workflow",
    "summarize_shadow_log",
    "temporary_parameter_pack",
    "update_pack_state",
]
