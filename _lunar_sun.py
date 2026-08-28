"""Calibrated lunar sun-elevation estimator.

Anchors the subsolar selenographic longitude to a known high-sun reference
(TMC-2 product with real elevation near local noon) and advances it at the
Moon's synodic rate (~12.189 deg/day over 29.53 days). Sub-solar latitude is
small (+-1.5 deg) and treated as 0 for ranking purposes.

Elevation(lon,lat,t) = asin( cos(lat) * cos(lon - lon_sub(t)) )
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from dataclasses import dataclass

SYNODIC_DAY = 29.530588853  # sidereal synodic month (days)
RATE = 360.0 / SYNODIC_DAY  # deg/day subsolar longitude advance


@dataclass
class Calibration:
    ref_time_jde: float
    ref_lon_east: float  # subsolar longitude at ref time (the high-sun site lon)

    def lon_sub(self, jde):
        return (self.ref_lon_east + (jde - self.ref_time_jde) * RATE) % 360.0

    def elevation(self, jde, site_lon_east, site_lat):
        lon_sub = self.lon_sub(jde)
        d2r = math.radians
        dlon = (site_lon_east - lon_sub) % 360.0
        if dlon > 180.0:
            dlon -= 360.0
        sinel = math.cos(d2r(site_lat)) * math.cos(d2r(dlon))
        return math.degrees(math.asin(max(-1.0, min(1.0, sinel))))


def jde_from_dt(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    y = dt.year; m = dt.month
    D = dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    if m <= 2:
        y -= 1; m += 12
    A = y // 100
    B = 2 - A + A // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + D + B - 1524.5


def dt_from_iso(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


if __name__ == "__main__":
    # Anchor: TMC 2025-07-07T18:53Z at site 25.2E elev 69.35 (high sun ~ noon).
    ref_jde = jde_from_dt(dt_from_iso("2025-07-07T18:53:05"))
    cal = Calibration(ref_jde, 25.2)

    known = [
        ("OHRC eq 2023", "2023-10-04T04:06:03", 25.2, -13.3, 9.49),
        ("OHRC west2022", "2022-03-20T21:33:02", -125.3, 5.4, 0.13),
        ("TMC 0713 5N", "2023-07-11T21:59:03", -125.3, 5.4, 65.05),
        ("TMC 0707 eq", "2025-07-07T18:53:05", 25.2, -13.3, 69.35),
        ("TMC 0813 r2", "2026-08-13T06:27:39", 141.5, -15.0, 50.94),
        ("OHRC sp2026 #1", "2026-01-03T06:09:04", 27.7, -85.3, 6.73),
    ]
    print("model (calibrated) vs real:")
    for name, t, lon, lat, real in known:
        est = cal.elevation(jde_from_dt(dt_from_iso(t)), lon, lat)
        print(f"  {name:16s} est {est:6.2f}  real {real:6.2f}  (lon_sub={cal.lon_sub(jde_from_dt(dt_from_iso(t))):.1f})")
