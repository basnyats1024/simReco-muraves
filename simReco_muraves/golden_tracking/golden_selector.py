"""Golden-track selection gate."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol


class GoldenRejectCode(IntEnum):
    ACCEPTED = 0
    INVALID_3P_XZ = 1
    INVALID_4P_XZ = 2


class GoldenEvent(Protocol):
    best_track_3p_chi_square_xz: float
    best_track_4p_chi_square_xz: float

    best_track_3p_xz_index: int
    best_track_3p_xy_index: int

    track_3p_of_4p_index_xz: int
    track_3p_of_4p_index_xy: int

    track_4p_index_xz: int
    track_4p_index_xy: int


@dataclass(frozen=True)
class GoldenSelection:
    accepted: bool

    best_3p_index_xz: int = -1
    best_3p_index_xy: int = -1

    parent_3p_index_xz: int = -1
    parent_3p_index_xy: int = -1

    extension_4p_index_xz: int = -1
    extension_4p_index_xy: int = -1

    reject_code: GoldenRejectCode = GoldenRejectCode.ACCEPTED


def select_golden_track(event: GoldenEvent) -> GoldenSelection:
    """Apply the main-reconstruction XZ-only golden-track gate."""

    if event.best_track_3p_chi_square_xz == -1:
        return GoldenSelection(
            accepted=False,
            reject_code=GoldenRejectCode.INVALID_3P_XZ,
        )

    if event.best_track_4p_chi_square_xz == -1:
        return GoldenSelection(
            accepted=False,
            reject_code=GoldenRejectCode.INVALID_4P_XZ,
        )

    return GoldenSelection(
        accepted=True,
        best_3p_index_xz=event.best_track_3p_xz_index,
        best_3p_index_xy=event.best_track_3p_xy_index,
        parent_3p_index_xz=event.track_3p_of_4p_index_xz,
        parent_3p_index_xy=event.track_3p_of_4p_index_xy,
        extension_4p_index_xz=event.track_4p_index_xz,
        extension_4p_index_xy=event.track_4p_index_xy,
        reject_code=GoldenRejectCode.ACCEPTED,
    )
