"""
Module: registry.py

Purpose:
    Implement registry utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from config.analytics_feature_policy import (
    DealerFlowPolicyConfig,
    FlowImbalancePolicyConfig,
    IvHvSpreadPolicyConfig,
    SmartMoneyFlowPolicyConfig,
    VolatilityRegimePolicyConfig,
)
from config.dealer_hedging_pressure_policy import DealerHedgingPressurePolicyConfig
from config.event_window_policy import EventWindowPolicyConfig
from config.gamma_vol_acceleration_policy import GammaVolAccelerationPolicyConfig
from config.global_risk_policy import GlobalRiskPolicyConfig
from config.large_move_policy import LARGE_MOVE_PROBABILITY_CONFIG
from config.news_keyword_policy import HEADLINE_RULES
from config.news_category_policy import (
    CATEGORY_GLOBAL_BIAS_MULTIPLIERS,
    CATEGORY_IMPACT_MULTIPLIERS,
    CATEGORY_INDIA_BIAS_MULTIPLIERS,
    CATEGORY_SENTIMENT_MULTIPLIERS,
    CATEGORY_VOL_MULTIPLIERS,
)
from config.option_efficiency_policy import OptionEfficiencyPolicyConfig
from config.probability_feature_policy import ProbabilityFeaturePolicyConfig
from config.signal_evaluation_scoring import (
    SIGNAL_EVALUATION_DIRECTION_WEIGHTS,
    SIGNAL_EVALUATION_SCORE_WEIGHTS,
    SIGNAL_EVALUATION_SELECTION_POLICY,
    SIGNAL_EVALUATION_THRESHOLDS,
    SIGNAL_EVALUATION_TIMING_WEIGHTS,
)
from config.signal_policy import (
    ActivationScorePolicyConfig,
    CONFIRMATION_FILTER_CONFIG,
    CONSENSUS_SCORE_CONFIG,
    DataQualityPolicyConfig,
    DIRECTION_MIN_MARGIN,
    DIRECTION_MIN_SCORE,
    DIRECTION_VOTE_WEIGHTS,
    ExecutionRegimePolicyConfig,
    LargeMoveScoringPolicyConfig,
    TRADE_RUNTIME_THRESHOLDS,
    TRADE_STRENGTH_WEIGHTS,
    TradeModifierPolicyConfig,
)
from config.strike_selection_policy import STRIKE_SELECTION_SCORE_CONFIG
from macro.macro_news_config import (
    HeadlineClassificationConfig,
    MacroNewsAdjustmentConfig,
    MacroNewsAggregationConfig,
    MacroNewsRegimeConfig,
)
from tuning.models import ParameterDefinition


GROUP_TUNING_METADATA = {
    "trade_strength": {
        "search_strategy": "coordinate_descent",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "low",
        "tuning_priority": 10,
    },
    "confirmation_filter": {
        "search_strategy": "coordinate_descent",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "low",
        "tuning_priority": 12,
    },
    "macro_news": {
        "search_strategy": "latin_hypercube",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "high",
        "tuning_priority": 20,
    },
    "global_risk": {
        "search_strategy": "latin_hypercube",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "high",
        "tuning_priority": 22,
    },
    "gamma_vol_acceleration": {
        "search_strategy": "latin_hypercube",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "high",
        "tuning_priority": 26,
    },
    "dealer_pressure": {
        "search_strategy": "latin_hypercube",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "high",
        "tuning_priority": 28,
    },
    "option_efficiency": {
        "search_strategy": "coordinate_descent",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "medium",
        "tuning_priority": 24,
    },
    "strike_selection": {
        "search_strategy": "latin_hypercube",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "high",
        "tuning_priority": 34,
    },
    "large_move_probability": {
        "search_strategy": "latin_hypercube",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "medium",
        "tuning_priority": 32,
    },
    "event_windows": {
        "search_strategy": "coordinate_descent",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "medium",
        "tuning_priority": 36,
    },
    "keyword_category": {
        "search_strategy": "latin_hypercube",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "high",
        "tuning_priority": 38,
    },
    "evaluation_thresholds": {
        "search_strategy": "coordinate_descent",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "high",
        "tuning_priority": 30,
    },
    "signal_engine": {
        "search_strategy": "coordinate_descent",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "medium",
        "tuning_priority": 16,
    },
    "analytics": {
        "search_strategy": "coordinate_descent",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "medium",
        "tuning_priority": 18,
    },
    "headline_rules": {
        "search_strategy": "latin_hypercube",
        "validation_mode": "walk_forward_regime_aware",
        "overfit_risk": "high",
        "tuning_priority": 40,
    },
}


class ParameterRegistry:
    """
    Purpose:
        Represent ParameterRegistry within the repository.
    
    Context:
        Used within the `registry` module. The class standardizes records that move through search, validation, shadow mode, and promotion.
    
    Attributes:
        None: The class primarily defines behavior or a protocol contract rather than stored fields.
    
    Notes:
        The class groups behavior and state that need to stay explicit for maintainability and auditability.
    """
    def __init__(self, definitions: list[ParameterDefinition]):
        """
        Purpose:
            Process init for downstream use.
        
        Context:
            Method on `ParameterRegistry` within the tuning layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            definitions (list[ParameterDefinition]): Input associated with definitions.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
        """
        self._definitions = {definition.key: definition for definition in definitions}

    def get(self, key: str) -> ParameterDefinition:
        """
        Purpose:
            Process get for downstream use.
        
        Context:
            Method on `ParameterRegistry` within the tuning layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            key (str): Input associated with key.
        
        Returns:
            ParameterDefinition: Result returned by the helper.
        
        Notes:
            The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
        """
        return self._definitions[key]

    def items(self):
        """
        Purpose:
            Process items for downstream use.
        
        Context:
            Method on `ParameterRegistry` within the tuning layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
        """
        return self._definitions.items()

    def keys(self):
        """
        Purpose:
            Process keys for downstream use.
        
        Context:
            Method on `ParameterRegistry` within the tuning layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            Any: Result returned by the helper.
        
        Notes:
            The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
        """
        return self._definitions.keys()

    def defaults(self) -> dict[str, Any]:
        """
        Purpose:
            Process defaults for downstream use.
        
        Context:
            Method on `ParameterRegistry` within the tuning layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            None: This helper does not require caller-supplied inputs.
        
        Returns:
            dict[str, Any]: Result returned by the helper.
        
        Notes:
            The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
        """
        return {
            key: definition.default_value
            for key, definition in self._definitions.items()
        }

    def serialize(self, current_values: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Purpose:
            Process serialize for downstream use.
        
        Context:
            Method on `ParameterRegistry` within the tuning layer. It keeps the object's contract explicit for downstream callers.
        
        Inputs:
            current_values (dict[str, Any] | None): Input associated with current values.
        
        Returns:
            dict[str, Any]: Result returned by the helper.
        
        Notes:
            The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
        """
        current_values = current_values or {}
        return {
            key: definition.to_dict(current_values.get(key))
            for key, definition in sorted(self._definitions.items())
        }


def _value_type_name(value: Any) -> str:
    """
    Purpose:
        Process value type name for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        value (Any): Input associated with value.
    
    Returns:
        str: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def _parameter_definition(
    *,
    key: str,
    module: str,
    group: str,
    category: str,
    default_value: Any,
    description: str,
    tunable: bool = True,
    live_safe: bool = True,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    allowed_values: tuple[Any, ...] | None = None,
) -> ParameterDefinition:
    """
    Purpose:
        Process parameter definition for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        key (str): Input associated with key.
        module (str): Input associated with module.
        group (str): Input associated with group.
        category (str): Input associated with category.
        default_value (Any): Input associated with default value.
        description (str): Input associated with description.
        tunable (bool): Boolean flag associated with tunable.
        live_safe (bool): Boolean flag associated with live_safe.
        min_value (float | int | None): Input associated with min value.
        max_value (float | int | None): Input associated with max value.
        allowed_values (tuple[Any, ...] | None): Input associated with allowed values.
    
    Returns:
        ParameterDefinition: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    metadata = GROUP_TUNING_METADATA.get(group, {})
    return ParameterDefinition(
        key=key,
        name=key.split(".")[-1],
        module=module,
        group=group,
        category=category,
        default_value=default_value,
        value_type=_value_type_name(default_value),
        description=description,
        tunable=tunable,
        live_safe=live_safe,
        min_value=min_value,
        max_value=max_value,
        allowed_values=allowed_values,
        search_strategy=str(metadata.get("search_strategy", "group_random_search")),
        validation_mode=str(metadata.get("validation_mode", "walk_forward_regime_aware")),
        overfit_risk=str(metadata.get("overfit_risk", "medium")),
        tuning_priority=int(metadata.get("tuning_priority", 50)),
        tune_as_group=True,
    )


def _from_mapping(
    *,
    prefix: str,
    module: str,
    group: str,
    category: str,
    mapping: dict[str, Any],
    description_prefix: str,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    live_safe: bool = True,
) -> list[ParameterDefinition]:
    """
    Purpose:
        Process from mapping for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        prefix (str): Input associated with prefix.
        module (str): Input associated with module.
        group (str): Input associated with group.
        category (str): Input associated with category.
        mapping (dict[str, Any]): Input associated with mapping.
        description_prefix (str): Input associated with description prefix.
        min_value (float | int | None): Input associated with min value.
        max_value (float | int | None): Input associated with max value.
        live_safe (bool): Boolean flag associated with live_safe.
    
    Returns:
        list[ParameterDefinition]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return [
        _parameter_definition(
            key=f"{prefix}.{name}",
            module=module,
            group=group,
            category=category,
            default_value=value,
            description=f"{description_prefix}: {name}",
            min_value=min_value,
            max_value=max_value,
            live_safe=live_safe,
        )
        for name, value in mapping.items()
    ]


def _from_dataclass(
    *,
    prefix: str,
    module: str,
    group: str,
    category: str,
    config_obj: Any,
    description_prefix: str,
    live_safe: bool = True,
    min_values: dict[str, float | int] | None = None,
    max_values: dict[str, float | int] | None = None,
) -> list[ParameterDefinition]:
    """
    Purpose:
        Process from dataclass for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        prefix (str): Input associated with prefix.
        module (str): Input associated with module.
        group (str): Input associated with group.
        category (str): Input associated with category.
        config_obj (Any): Input associated with config obj.
        description_prefix (str): Input associated with description prefix.
        live_safe (bool): Boolean flag associated with live_safe.
    
    Returns:
        list[ParameterDefinition]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    definitions = []
    for field in fields(config_obj):
        value = getattr(config_obj, field.name)
        definitions.append(
            _parameter_definition(
                key=f"{prefix}.{field.name}",
                module=module,
                group=group,
                category=category,
                default_value=value,
                description=f"{description_prefix}: {field.name}",
                live_safe=live_safe,
                min_value=(min_values or {}).get(field.name),
                max_value=(max_values or {}).get(field.name),
            )
        )
    return definitions


def build_default_parameter_registry() -> ParameterRegistry:
    """
    Purpose:
        Build the default parameter registry used by downstream components.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        ParameterRegistry: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    definitions: list[ParameterDefinition] = []

    definitions.extend(
        _from_mapping(
            prefix="trade_strength.direction_vote",
            module="config.signal_policy",
            group="trade_strength",
            category="direction_vote",
            mapping=DIRECTION_VOTE_WEIGHTS,
            description_prefix="Direction vote weight",
            min_value=-5.0,
            max_value=5.0,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="trade_strength.scoring",
            module="config.signal_policy",
            group="trade_strength",
            category="scoring",
            mapping=TRADE_STRENGTH_WEIGHTS,
            description_prefix="Trade strength scoring weight",
            min_value=-30,
            max_value=30,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="trade_strength.consensus",
            module="config.signal_policy",
            group="trade_strength",
            category="consensus",
            mapping=CONSENSUS_SCORE_CONFIG,
            description_prefix="Directional consensus score",
            min_value=-20,
            max_value=20,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="trade_strength.runtime_thresholds",
            module="config.signal_policy",
            group="trade_strength",
            category="runtime_thresholds",
            mapping=TRADE_RUNTIME_THRESHOLDS,
            description_prefix="Trade runtime threshold",
            min_value=0,
            max_value=100,
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="signal_engine.data_quality",
            module="config.signal_policy",
            group="signal_engine",
            category="data_quality",
            config_obj=DataQualityPolicyConfig(),
            description_prefix="Signal-engine data-quality parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="signal_engine.execution_regime",
            module="config.signal_policy",
            group="signal_engine",
            category="execution_regime",
            config_obj=ExecutionRegimePolicyConfig(),
            description_prefix="Signal-engine execution-regime parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="signal_engine.large_move_scoring",
            module="config.signal_policy",
            group="signal_engine",
            category="large_move_scoring",
            config_obj=LargeMoveScoringPolicyConfig(),
            description_prefix="Large-move trade-strength parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="signal_engine.trade_modifiers",
            module="config.signal_policy",
            group="signal_engine",
            category="trade_modifiers",
            config_obj=TradeModifierPolicyConfig(),
            description_prefix="Signal-engine trade-modifier parameter",
        )
    )
    definitions.extend(
        [
            _parameter_definition(
                key=f"signal_engine.probability.{field.name}",
                module="config.probability_feature_policy",
                group="signal_engine",
                category="probability",
                default_value=getattr(ProbabilityFeaturePolicyConfig(), field.name),
                description=f"Signal-engine probability parameter: {field.name}",
                min_value={
                    "vacuum_breakout_strength": 0.0,
                    "vacuum_near_strength": 0.0,
                    "vacuum_watch_strength": 0.0,
                    "vacuum_default_strength": 0.0,
                    "vacuum_gap_pct_cap": 0.1,
                    "vacuum_gap_base_weight": 0.0,
                    "vacuum_gap_proximity_weight": 0.0,
                    "vacuum_void_count_cap": 0,
                    "vacuum_void_increment": 0.0,
                    "hedging_bias_upside_acceleration_score": -1.0,
                    "hedging_bias_downside_acceleration_score": -1.0,
                    "hedging_bias_upside_pinning_score": -1.0,
                    "hedging_bias_downside_pinning_score": -1.0,
                    "hedging_bias_pinning_score": -1.0,
                    "smart_money_bullish_score": -1.0,
                    "smart_money_bearish_score": -1.0,
                    "smart_money_neutral_score": -1.0,
                    "smart_money_categorical_weight": 0.0,
                    "smart_money_flow_imbalance_weight": 0.0,
                    "intraday_range_anchor_multiplier": 0.5,
                    "intraday_range_baseline_floor_pct": 0.05,
                    "intraday_range_denominator_floor_pct": 0.05,
                    "intraday_range_clip_cap": 0.1,
                    "atm_iv_low": 1.0,
                    "atm_iv_high": 1.0,
                    "probability_default_rule": 0.0,
                    "probability_floor": 0.0,
                    "probability_ceiling": 0.0,
                    "probability_rule_weight": 0.0,
                    "probability_ml_weight": 0.0,
                    "probability_intercept": -1.0,
                    "probability_scale": 0.0,
                    "categorical_flow_weight": 0.0,
                    "smart_money_flow_weight": 0.0,
                    "calibration_midpoint": 0.0,
                    "calibration_steepness": 0.1,
                }.get(field.name),
                max_value={
                    "vacuum_breakout_strength": 1.0,
                    "vacuum_near_strength": 1.0,
                    "vacuum_watch_strength": 1.0,
                    "vacuum_default_strength": 1.0,
                    "vacuum_gap_pct_cap": 5.0,
                    "vacuum_gap_base_weight": 1.0,
                    "vacuum_gap_proximity_weight": 1.0,
                    "vacuum_void_count_cap": 20,
                    "vacuum_void_increment": 0.2,
                    "hedging_bias_upside_acceleration_score": 1.0,
                    "hedging_bias_downside_acceleration_score": 1.0,
                    "hedging_bias_upside_pinning_score": 1.0,
                    "hedging_bias_downside_pinning_score": 1.0,
                    "hedging_bias_pinning_score": 1.0,
                    "smart_money_bullish_score": 1.0,
                    "smart_money_bearish_score": 1.0,
                    "smart_money_neutral_score": 1.0,
                    "smart_money_categorical_weight": 1.0,
                    "smart_money_flow_imbalance_weight": 1.0,
                    "intraday_range_anchor_multiplier": 5.0,
                    "intraday_range_baseline_floor_pct": 5.0,
                    "intraday_range_denominator_floor_pct": 5.0,
                    "intraday_range_clip_cap": 5.0,
                    "atm_iv_low": 80.0,
                    "atm_iv_high": 80.0,
                    "probability_default_rule": 1.0,
                    "probability_floor": 1.0,
                    "probability_ceiling": 1.0,
                    "probability_rule_weight": 1.0,
                    "probability_ml_weight": 1.0,
                    "probability_intercept": 1.0,
                    "probability_scale": 2.0,
                    "categorical_flow_weight": 1.0,
                    "smart_money_flow_weight": 1.0,
                    "calibration_midpoint": 1.0,
                    "calibration_steepness": 10.0,
                }.get(field.name),
            )
            for field in fields(ProbabilityFeaturePolicyConfig())
        ]
    )
    definitions.extend(
        _from_dataclass(
            prefix="signal_engine.activation_score",
            module="config.signal_policy",
            group="signal_engine",
            category="activation_score",
            config_obj=ActivationScorePolicyConfig(),
            description_prefix="Activation-score setup-readiness parameter",
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="confirmation_filter.core",
            module="config.signal_policy",
            group="confirmation_filter",
            category="core",
            mapping=CONFIRMATION_FILTER_CONFIG,
            description_prefix="Confirmation filter parameter",
            min_value=-10,
            max_value=10,
        )
    )
    definitions.extend(
        [
            _parameter_definition(
                key="trade_strength.direction_thresholds.min_score",
                module="config.signal_policy",
                group="trade_strength",
                category="direction_thresholds",
                default_value=DIRECTION_MIN_SCORE,
                description="Minimum directional vote score required",
                min_value=0.0,
                max_value=10.0,
            ),
            _parameter_definition(
                key="trade_strength.direction_thresholds.min_margin",
                module="config.signal_policy",
                group="trade_strength",
                category="direction_thresholds",
                default_value=DIRECTION_MIN_MARGIN,
                description="Minimum directional vote margin required",
                min_value=0.0,
                max_value=10.0,
            ),
        ]
    )

    definitions.extend(
        _from_dataclass(
            prefix="analytics.flow_imbalance",
            module="config.analytics_feature_policy",
            group="analytics",
            category="flow_imbalance",
            config_obj=FlowImbalancePolicyConfig(),
            description_prefix="Analytics flow-imbalance parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="analytics.smart_money_flow",
            module="config.analytics_feature_policy",
            group="analytics",
            category="smart_money_flow",
            config_obj=SmartMoneyFlowPolicyConfig(),
            description_prefix="Analytics smart-money-flow parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="analytics.volatility_regime",
            module="config.analytics_feature_policy",
            group="analytics",
            category="volatility_regime",
            config_obj=VolatilityRegimePolicyConfig(),
            description_prefix="Analytics volatility-regime parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="analytics.iv_hv_spread",
            module="config.analytics_feature_policy",
            group="analytics",
            category="iv_hv_spread",
            config_obj=IvHvSpreadPolicyConfig(),
            description_prefix="Analytics IV-HV spread parameter",
            min_values={
                "rich_threshold_relative": 0.02,
                "cheap_threshold_relative": -0.60,
            },
            max_values={
                "rich_threshold_relative": 0.60,
                "cheap_threshold_relative": -0.02,
            },
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="analytics.dealer_flow",
            module="config.analytics_feature_policy",
            group="analytics",
            category="dealer_flow",
            config_obj=DealerFlowPolicyConfig(),
            description_prefix="Analytics dealer-flow weight parameter",
            min_values={
                "gamma_weight": 0.0,
                "charm_weight": 0.0,
            },
            max_values={
                "gamma_weight": 2.0,
                "charm_weight": 2.0,
            },
        )
    )

    definitions.extend(
        _from_dataclass(
            prefix="macro_news.headline_classification",
            module="macro.macro_news_config",
            group="macro_news",
            category="headline_classification",
            config_obj=HeadlineClassificationConfig(),
            description_prefix="Headline classification parameter",
            live_safe=False,
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="macro_news.aggregation",
            module="macro.macro_news_config",
            group="macro_news",
            category="aggregation",
            config_obj=MacroNewsAggregationConfig(),
            description_prefix="Macro news aggregation parameter",
            live_safe=False,
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="macro_news.regime",
            module="macro.macro_news_config",
            group="macro_news",
            category="regime",
            config_obj=MacroNewsRegimeConfig(),
            description_prefix="Macro news regime parameter",
            live_safe=False,
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="macro_news.adjustment",
            module="macro.macro_news_config",
            group="macro_news",
            category="adjustment",
            config_obj=MacroNewsAdjustmentConfig(),
            description_prefix="Macro news engine adjustment parameter",
            min_values={
                "risk_off_call_size_high_vol": 0.25,
                "risk_off_call_size_normal": 0.40,
                "risk_off_put_size_medium_vol": 0.50,
                "risk_on_call_size_medium_vol": 0.50,
                "risk_on_put_size_high_vol": 0.25,
                "risk_on_put_size_normal": 0.40,
                "generic_high_vol_size_cap": 0.30,
                "generic_medium_vol_size_cap": 0.50,
            },
            max_values={
                "risk_off_call_size_high_vol": 0.85,
                "risk_off_call_size_normal": 1.00,
                "risk_off_put_size_medium_vol": 1.00,
                "risk_on_call_size_medium_vol": 1.00,
                "risk_on_put_size_high_vol": 0.85,
                "risk_on_put_size_normal": 1.00,
                "generic_high_vol_size_cap": 0.90,
                "generic_medium_vol_size_cap": 1.00,
            },
        )
    )
    definitions.extend(
        [
            _parameter_definition(
                key=f"headline_rules.{rule['name']}.{field_name}",
                module="config.news_keyword_policy",
                group="headline_rules",
                category=rule["name"],
                default_value=rule[field_name],
                description=f"Headline rule scalar parameter: {rule['name']}.{field_name}",
                min_value=-1.5 if field_name != "impact_score" else 0.0,
                max_value=100.0 if field_name == "impact_score" else 1.5,
                live_safe=False,
            )
            for rule in HEADLINE_RULES
            for field_name in (
                "sentiment_weight",
                "vol_weight",
                "impact_score",
                "india_macro_bias",
                "global_risk_bias",
            )
        ]
    )

    definitions.extend(
        _from_dataclass(
            prefix="global_risk.core",
            module="config.global_risk_policy",
            group="global_risk",
            category="core",
            config_obj=GlobalRiskPolicyConfig(),
            description_prefix="Global risk policy parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="gamma_vol_acceleration.core",
            module="config.gamma_vol_acceleration_policy",
            group="gamma_vol_acceleration",
            category="core",
            config_obj=GammaVolAccelerationPolicyConfig(),
            description_prefix="Gamma-vol acceleration parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="dealer_pressure.core",
            module="config.dealer_hedging_pressure_policy",
            group="dealer_pressure",
            category="core",
            config_obj=DealerHedgingPressurePolicyConfig(),
            description_prefix="Dealer hedging pressure parameter",
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="option_efficiency.core",
            module="config.option_efficiency_policy",
            group="option_efficiency",
            category="core",
            config_obj=OptionEfficiencyPolicyConfig(),
            description_prefix="Option efficiency parameter",
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="strike_selection.core",
            module="config.strike_selection_policy",
            group="strike_selection",
            category="core",
            mapping=STRIKE_SELECTION_SCORE_CONFIG,
            description_prefix="Strike selection parameter",
            min_value=-100.0,
            max_value=1000000.0,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="large_move_probability.core",
            module="config.large_move_policy",
            group="large_move_probability",
            category="core",
            mapping=LARGE_MOVE_PROBABILITY_CONFIG,
            description_prefix="Large-move probability parameter",
            min_value=-1.0,
            max_value=1.0,
        )
    )
    definitions.extend(
        _from_dataclass(
            prefix="event_windows.core",
            module="config.event_window_policy",
            group="event_windows",
            category="core",
            config_obj=EventWindowPolicyConfig(),
            description_prefix="Event window parameter",
            live_safe=False,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="keyword_category.sentiment",
            module="config.news_category_policy",
            group="keyword_category",
            category="sentiment",
            mapping=CATEGORY_SENTIMENT_MULTIPLIERS,
            description_prefix="Keyword category sentiment multiplier",
            min_value=0.5,
            max_value=1.5,
            live_safe=False,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="keyword_category.volatility",
            module="config.news_category_policy",
            group="keyword_category",
            category="volatility",
            mapping=CATEGORY_VOL_MULTIPLIERS,
            description_prefix="Keyword category volatility multiplier",
            min_value=0.5,
            max_value=1.5,
            live_safe=False,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="keyword_category.impact",
            module="config.news_category_policy",
            group="keyword_category",
            category="impact",
            mapping=CATEGORY_IMPACT_MULTIPLIERS,
            description_prefix="Keyword category impact multiplier",
            min_value=0.5,
            max_value=1.5,
            live_safe=False,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="keyword_category.india_bias",
            module="config.news_category_policy",
            group="keyword_category",
            category="india_bias",
            mapping=CATEGORY_INDIA_BIAS_MULTIPLIERS,
            description_prefix="Keyword category India bias multiplier",
            min_value=0.5,
            max_value=1.5,
            live_safe=False,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="keyword_category.global_bias",
            module="config.news_category_policy",
            group="keyword_category",
            category="global_bias",
            mapping=CATEGORY_GLOBAL_BIAS_MULTIPLIERS,
            description_prefix="Keyword category global-risk bias multiplier",
            min_value=0.5,
            max_value=1.5,
            live_safe=False,
        )
    )

    definitions.extend(
        _from_mapping(
            prefix="evaluation_thresholds.score_weights",
            module="config.signal_evaluation_scoring",
            group="evaluation_thresholds",
            category="score_weights",
            mapping=SIGNAL_EVALUATION_SCORE_WEIGHTS,
            description_prefix="Signal evaluation composite weight",
            min_value=0.0,
            max_value=1.0,
            live_safe=False,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="evaluation_thresholds.direction_weights",
            module="config.signal_evaluation_scoring",
            group="evaluation_thresholds",
            category="direction_weights",
            mapping=SIGNAL_EVALUATION_DIRECTION_WEIGHTS,
            description_prefix="Signal evaluation direction weight",
            min_value=0.0,
            max_value=5.0,
            live_safe=False,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="evaluation_thresholds.timing_weights",
            module="config.signal_evaluation_scoring",
            group="evaluation_thresholds",
            category="timing_weights",
            mapping=SIGNAL_EVALUATION_TIMING_WEIGHTS,
            description_prefix="Signal evaluation timing weight",
            min_value=0.0,
            max_value=5.0,
            live_safe=False,
        )
    )
    definitions.extend(
        _from_mapping(
            prefix="evaluation_thresholds.core",
            module="config.signal_evaluation_scoring",
            group="evaluation_thresholds",
            category="core",
            mapping=SIGNAL_EVALUATION_THRESHOLDS,
            description_prefix="Signal evaluation threshold",
            min_value=0.0,
            max_value=10.0,
            live_safe=False,
        )
    )
    definitions.extend(
        [
            _parameter_definition(
                key=f"evaluation_thresholds.selection.{name}",
                module="config.signal_evaluation_scoring",
                group="evaluation_thresholds",
                category="selection",
                default_value=value,
                description=f"Dataset experiment selection threshold: {name}",
                min_value={
                    "trade_strength_floor": 0.0,
                    "composite_signal_score_floor": 0.0,
                    "tradeability_score_floor": 0.0,
                    "move_probability_floor": 0.0,
                    "option_efficiency_score_floor": 0.0,
                    "global_risk_score_cap": 0.0,
                }.get(name),
                max_value={
                    "trade_strength_floor": 100.0,
                    "composite_signal_score_floor": 100.0,
                    "tradeability_score_floor": 100.0,
                    "move_probability_floor": 1.0,
                    "option_efficiency_score_floor": 100.0,
                    "global_risk_score_cap": 100.0,
                }.get(name),
                live_safe=False,
            )
            for name, value in SIGNAL_EVALUATION_SELECTION_POLICY.items()
        ]
    )

    return ParameterRegistry(definitions)


_REGISTRY: ParameterRegistry | None = None


def get_parameter_registry() -> ParameterRegistry:
    """
    Purpose:
        Return parameter registry for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        None: This helper does not require caller-supplied inputs.
    
    Returns:
        ParameterRegistry: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_default_parameter_registry()
    return _REGISTRY
