"""Trajectory Winter-Shapley attribution for the transformer conflict model.

Modules
-------
config     : AttributionConfig dataclasses + YAML loader
values     : ConflictValueFn — batch evaluator of v(S) for one event
winter_mc  : nested-permutation MC engine (Winter value) + exact Shapley
outputs    : level aggregation, JSON/CSV serialization
driver     : sample selection, checkpoint loading, per-event loop
visualize  : matplotlib rendering of per-event results (optional dependency)

See 04_轨迹WinterShapley归因方案.md for the game definition and the two
readouts (symmetric "evidence weight" vs. chronological "risk revelation").
"""
