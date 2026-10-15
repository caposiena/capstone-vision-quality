from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionProfile:
    key: str
    label: str
    roi_left: float
    roi_right: float
    roi_top: float
    roi_bottom: float
    deadline_x: float
    direction: str

    def as_percentages(self) -> dict[str, int]:
        return {
            "left": round(self.roi_left * 100),
            "right": round(self.roi_right * 100),
            "top": round(self.roi_top * 100),
            "bottom": round(self.roi_bottom * 100),
            "deadline": round(self.deadline_x * 100),
        }


PROFILES: dict[str, AcquisitionProfile] = {
    "standard_rtl": AcquisitionProfile(
        key="standard_rtl",
        label="Standard A - destra verso sinistra",
        roi_left=0.30,
        roi_right=0.12,
        roi_top=0.20,
        roi_bottom=0.25,
        deadline_x=0.35,
        direction="rtl",
    ),
    "zoom_ltr": AcquisitionProfile(
        key="zoom_ltr",
        label="Standard B - sinistra verso destra / zoom",
        roi_left=0.12,
        roi_right=0.30,
        roi_top=0.18,
        roi_bottom=0.24,
        deadline_x=0.65,
        direction="ltr",
    ),
}

DEFAULT_PROFILE_KEY = "standard_rtl"


def get_profile(key: str = DEFAULT_PROFILE_KEY) -> AcquisitionProfile:
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"Profilo acquisizione sconosciuto: {key}") from exc
