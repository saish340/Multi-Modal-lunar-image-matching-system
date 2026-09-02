"""Pure-Python DBF parser used to dump PRADAN footprint catalog rows.

DBF (dBase III) layout:
  byte 0        : version
  bytes 4-7     : record count (int32 LE)
  bytes 8-9     : header size (int16 LE)
  bytes 10-11   : record size (int16 LE)
  from byte 32  : 32-byte field descriptors, terminated by 0x0D
  field desc    : name[11], type[1], ..., length at +16, decimals at +17
  records       : 1 deletion byte + concatenated fixed-width fields
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def read_dbf(path: str | Path) -> list[dict[str, str]]:
    data = Path(path).read_bytes()
    n_records = struct.unpack_from("<i", data, 4)[0]
    header_size = struct.unpack_from("<h", data, 8)[0]
    record_size = struct.unpack_from("<h", data, 10)[0]

    fields: list[tuple[str, int]] = []
    offset = 32
    while data[offset] != 0x0D:
        name = data[offset:offset + 11].split(b"\x00")[0].decode("ascii", "ignore")
        ftype = chr(data[offset + 11])
        flen = data[offset + 16]
        fields.append((f"{name}:{ftype}", flen))
        offset += 32

    records: list[dict[str, str]] = []
    pos = header_size
    for _ in range(n_records):
        rec = data[pos:pos + record_size]
        if len(rec) < record_size:
            break
        pos += record_size
        if rec[0:1] in (b"*", b"\x1a"):
            continue  # deleted record
        out: dict[str, str] = {}
        cur = 1
        for (name, flen) in fields:
            raw = rec[cur:cur + flen]
            cur += flen
            out[name] = raw.decode("ascii", "ignore").strip()
        records.append(out)
    return records


def search(dbf_path: str | Path, needle: str) -> list[dict[str, str]]:
    rows = read_dbf(dbf_path)
    hits = []
    for r in rows:
        joined = " ".join(str(v) for v in r.values())
        if needle.lower() in joined.lower():
            hits.append(r)
    return hits


def dump(rows: list[dict[str, str]], title: str) -> None:
    print(f"--- {title}: {len(rows)} hit(s) ---")
    for r in rows:
        for k, v in r.items():
            print(f"  {k:22s} = {v}")
        print()


if __name__ == "__main__":
    base = Path(r"d:\Multi-Modal-lunar-image-matching-system")
    ohrc_dbfs = list((base / "OHRC_ShapeFiles").rglob("*.dbf"))
    tmc_dbfs = list((base / "TMC2_ShapeFiles").rglob("*.dbf"))

    needles = [
        "20200827T0030107497",  # OHRC #1
        "20200827T0226453039",  # OHRC #2
        "20231101T0125121377",  # TMC-2 ortho
    ]

    for dbf in sorted(ohrc_dbfs) + sorted(tmc_dbfs):
        for nd in needles:
            try:
                hits = search(dbf, nd)
            except Exception as exc:  # noqa: BLE001
                print(f"!! {dbf.name}: {exc}")
                continue
            if hits:
                dump(hits, f"{dbf.parent.name}/{dbf.name} ~ {nd}")
