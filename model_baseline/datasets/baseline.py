"""Baseline dataset form — multi-agent history → conflict labels.

TODO: implement CSV join / tensor packing. Left empty by design for now.
"""

from __future__ import annotations

from typing import Any, Optional


class BaselineConflictDataset:
    """
    Expected sample contract (for matching BaselineConflictModel):

    Variable neighbor count is handled by **fixed-A padding + agent_mask**:
      - Always emit A = 7 slots (ego + 6 azimuth roles).
      - Missing neighbors: zero-fill that slot's track, set agent_mask[a] = False.
      - Model mean-pools / softmax only over True slots.

    __getitem__ → dict with:
      - "x":           FloatTensor [A, T, F]  multi-agent history
      - "agent_mask":  BoolTensor  [A]        True if slot has a vehicle
      - "is_conflict": Float/Long scalar     0/1
      - "target_idx":  Long scalar           neighbor slot index in {0..A-2}
                                              (maps to roles[1:]); -1 if none
      - "event_id":    str / int             for debugging

    Agent axis A is ordered as:
      [ego, front, rear, left_front, left_rear, right_front, right_rear]
    Features F (suggested): [dx, dy, heading_rel, speed] relative to ego@t0.
    """

    def __init__(
        self,
        data_path: str,
        label_path: str,
        split: str = "train",
        future_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.future_path = future_path
        # TODO: load CSVs, group by Event_id, build index
        raise NotImplementedError("BaselineConflictDataset not implemented yet")

    def __len__(self) -> int:
        # TODO
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        # TODO: return tensors matching the contract above
        raise NotImplementedError
