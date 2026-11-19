"""The dashboard's aid inputs from the team's two announcement-based projection workbooks, one folder per data source.

    IHME   IHME_aid_cut_projections_announcements.xlsx   2023 IHME DAH (95 recipients)
    OECD   OECD_aid_cut_projections.xlsx                 2023 OECD CRS gross health ODA (154 recipients)

Both workbooks apply the same announcement rules and settings (US MOUs, Global Fund GC7/GC8, Gavi pledges, UK/AU
country allocations, exits, the German bilateral health exit, the Dutch budget, each donor's OECD ODA outlook for the
rest) to every recipient x original donor x channel row of their own 2023 baseline (sheet Base_Projection). Each
row's yearly cut is 1 - Proj <year> / Baseline. The model needs that money by service line, which neither workbook
has, so each row is split as follows:

  IHME rows  the row's own 2023 IHME program areas (country_data/, the same rows the workbook was built from; checked
             to match the workbook's DAH 2023 to the dollar), mapped to service lines with prepare_model.LINES.
  OECD rows  disease $ = Baseline x the row's HIV / TB / malaria share; split into service lines with the country's
             IHME 2023 mix for the same donor group and channel group (else the channel group, else the country, else
             all countries). Immunization = the row's newborn & child health $ x IHME's vaccine share of non-systems
             child health money (OECD reports health systems as its own area, so the systems part is already out;
             for Gavi this share is 100%).

Writes, for each source, model_data/sources/<source>/:
  dah_lines.csv          baseline cells (country x donor group x channel group x service line, US$ 2023) for the
                         dashboard countries, with each cell's own yearly cuts (cut_, known_cut_, mou_pred_cut_<year>)
  announcement_cuts.csv  every health area, every workbook recipient, by country x donor group x channel group
                         (base_kusd and the same cut columns): the money headline, known vs predicted, biggest risks
known_cut = the part of the cut from rules that apply a published number; mou_pred_cut = US bilateral money in a
country without an America First MOU (rule "US non-MOU"). Flows the workbooks leave out (debt-funded lending, China
and other non-DAC donors, Other / unallocable) are not in either baseline.

Usage (from the repo root):  python announcement_model/workbook_inputs.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from prepare_model import DASHBOARD_ISO3, LINES, channel_group, source_group  # noqa: E402

WORKBOOKS = {"ihme": REPO / "IHME_aid_cut_projections_announcements.xlsx",
             "oecd": REPO / "OECD_aid_cut_projections.xlsx"}
SRC_COL = {"ihme": "IHME source (original donor)", "oecd": "Source (original donor)"}
OUT = REPO / "model_data" / "sources"
YRS = range(2026, 2031)
KNOWN = {"US MOU", "Global Fund", "Gavi", "Zero", "Exit", "UK country allocation", "AU country allocation",
         "DE bilateral health exit", "NL health", "NL UNICEF", "WHO own"}
OWN = "Multilateral own funds"            # OECD only: IDA, EU, WHO, UNICEF, UNFPA, UNAIDS spending their own money
ISO_FIX = {"XKV": "KSV"}
LN = list(LINES)
BKT = {l: b for l, (b, _) in LINES.items()}
AREA = {"HIV": "Base share: HIV/AIDS", "TB": "Base share: Tuberculosis", "Malaria": "Base share: Malaria"}
NCH = "Base share: Newborn & child health"
NCH_HSS = ["nch_hss_other", "nch_hss_hrh", "nch_hss_me"]


def raw_ihme_2023() -> pd.DataFrame:
    """IHME 2023 by country x source x channel: US$ per service line, child health (non-systems) and the total."""
    raw = pd.concat(pd.read_csv(f, na_values=["-"], low_memory=False) for f in glob.glob(str(REPO / "country_data/*.csv")))
    raw = raw[raw.year == 2023]
    raw = raw.fillna({c: 0 for c in raw.columns if c.endswith("_dah_23")})
    for line, (_, cols) in LINES.items():
        raw[line] = raw[[f"{c}_dah_23" for c in cols]].sum(axis=1) * 1e3
    raw["nch_core"] = (raw["nch_dah_23"] - raw[[f"{c}_dah_23" for c in NCH_HSS]].sum(axis=1)) * 1e3
    raw["total"] = raw["dah_23"]                                      # thousands, like the workbook
    r = raw.groupby(["recipient_isocode", "source", "channel"])[LN + ["nch_core", "total"]].sum().reset_index()
    r = r.rename(columns={"recipient_isocode": "iso3"})
    r["src_grp"], r["chan_grp"] = r.source.map(source_group), r.channel.map(channel_group)
    return r


def load_rows(source: str) -> pd.DataFrame:
    bp = pd.read_excel(WORKBOOKS[source], "Base_Projection")
    bp["iso3"] = bp["Recipient ISO3"].replace(ISO_FIX)
    bp["source"], bp["channel"] = bp[SRC_COL[source]], bp["Channel"]
    own = bp["source"].str.contains("own funds", na=False)
    bp["src_grp"] = np.where(own, OWN, bp["source"].map(source_group))
    bp["chan_grp"] = bp["channel"].map(lambda c: "Other" if c in ("MULTI_EARMARKED", "PPP") else channel_group(c))
    bp = bp[bp.Baseline > 0].copy()
    for y in YRS:
        lost = bp.Baseline - bp[f"Proj {y}"]
        bp[f"l{y}"], bp[f"k{y}"] = lost, lost.where(bp.Rule.isin(KNOWN), 0.0)
        bp[f"p{y}"] = lost.where(bp.Rule == "US non-MOU", 0.0)
    return bp


def cut_table(bp: pd.DataFrame) -> pd.DataFrame:
    """All health areas by country x donor group x channel group: base_kusd and yearly cut shares."""
    g = bp.groupby(["iso3", "src_grp", "chan_grp"])[["Baseline"] + [f"{p}{y}" for p in "lkp" for y in YRS]].sum()
    g = g[g.Baseline > 0]
    out = pd.DataFrame({"base_kusd": g.Baseline})
    for part, p in (("cut", "l"), ("known_cut", "k"), ("mou_pred_cut", "p")):
        for y in YRS:
            out[f"{part}_{y}"] = (g[f"{p}{y}"] / g.Baseline).round(6)
    return out.reset_index()


def split_ihme(bp: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Each IHME row's service-line US$ from its own 2023 program areas."""
    m = bp.merge(raw[["iso3", "source", "channel"] + LN + ["total"]], on=["iso3", "source", "channel"], how="left")
    m[LN + ["total"]] = m[LN + ["total"]].fillna(0.0)
    d = m.loc[m.iso3.isin(DASHBOARD_ISO3)]
    bad = d[(d.total - d.Baseline).abs() > 1.5]
    assert bad.empty, f"workbook rows that don't match IHME 2023:\n{bad[['iso3', 'source', 'channel', 'Baseline', 'total']]}"
    return m


def split_oecd(bp: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Each OECD row's service-line US$: its disease shares, split by the closest IHME 2023 mix."""
    levels = [["iso3", "src_grp", "chan_grp"], ["iso3", "chan_grp"], ["iso3"]]
    mixes = [raw.groupby(k)[LN + ["nch_core"]].sum() for k in levels]
    world = raw[LN + ["nch_core"]].sum()

    def mix(row, cols):
        for k, m in zip(levels, mixes):
            key = tuple(row[c] for c in k)
            key = key[0] if len(key) == 1 else key
            if key in m.index:
                v = m.loc[key, cols].astype(float).clip(lower=0)     # IHME has a few negative adjustments
                if v.sum() > 0:
                    return v
        return world[cols].astype(float).clip(lower=0)

    out = bp.copy()
    for l in LN:
        out[l] = 0.0
    for i, x in bp.iterrows():
        for b, col in AREA.items():
            amt = x.Baseline * 1e3 * x[col]
            if amt > 0:
                v = mix(x, [l for l in LN if BKT[l] == b])
                for l, w in (v / v.sum()).items():
                    out.at[i, l] = amt * w
        nch = x.Baseline * 1e3 * x[NCH]
        if nch > 0:
            v = mix(x, ["imm", "nch_core"])
            out.at[i, "imm"] = nch * (min(1.0, v["imm"] / v["nch_core"]) if v["nch_core"] > 0 else 0.0)
    return out


def line_cells(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows with service-line US$ -> model cells (country x donor group x channel group x line) with yearly cuts."""
    rows = rows[rows.iso3.isin(DASHBOARD_ISO3)]
    parts = []
    for l in LN:
        d = rows[["iso3", "src_grp", "chan_grp"]].copy()
        share = (rows[l] / (rows.Baseline * 1e3)).to_numpy()
        d["usd"] = rows[l].to_numpy()
        for p in "lkp":
            for y in YRS:
                d[f"{p}{y}"] = rows[f"{p}{y}"].to_numpy() * 1e3 * share     # this line's part of the row's loss
        d["line"], d["bucket"] = l, BKT[l]
        parts.append(d[d.usd > 0])
    c = pd.concat(parts).groupby(["iso3", "src_grp", "chan_grp", "line", "bucket"], as_index=False).sum()
    for part, p in (("cut", "l"), ("known_cut", "k"), ("mou_pred_cut", "p")):
        for y in YRS:
            c[f"{part}_{y}"] = (c.pop(f"{p}{y}") / c.usd).round(6)
    return c


def build(source: str, raw: pd.DataFrame) -> dict:
    bp = load_rows(source)
    rows = split_ihme(bp, raw) if source == "ihme" else split_oecd(bp, raw)
    cells, cuts = line_cells(rows), cut_table(bp)
    d = OUT / source
    d.mkdir(parents=True, exist_ok=True)
    cells.to_csv(d / "dah_lines.csv", index=False)
    cuts.to_csv(d / "announcement_cuts.csv", index=False)
    missing = sorted(set(DASHBOARD_ISO3) - set(cells.iso3))
    return {"source": source, "cells": len(cells), "countries": cells.iso3.nunique(), "missing": missing,
            "base_4_diseases_B": cells.usd.sum() / 1e9,
            "by_bucket_M": (cells.groupby("bucket").usd.sum() / 1e6).round(0).to_dict(),
            "all_health_dashboard_B": cuts.loc[cuts.iso3.isin(DASHBOARD_ISO3), "base_kusd"].sum() / 1e6}


if __name__ == "__main__":
    raw = raw_ihme_2023()
    for s in WORKBOOKS:
        print(build(s, raw))
