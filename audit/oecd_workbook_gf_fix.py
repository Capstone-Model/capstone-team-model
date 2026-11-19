"""
Write the Global Fund disease split into the team's OECD projection workbook itself (Nov 2026).

    python audit/oecd_workbook_gf_fix.py

OECD_aid_cut_projections.xlsx, sheet Base_Projection: for every row whose Channel name is "Global Fund" and whose
recipient has a 2023 Global Fund budget (audit/logic_review/gf_disease_split.csv), the HIV/AIDS, Malaria and
Tuberculosis shares are re-divided by the Global Fund's own split, keeping their sum. Both the base shares (columns
AC, AD, AE) and the projected shares (AL, AM, AN). Every other cell is untouched.

How: the cells are typed numbers, so only their <v> values are rewritten inside the .xlsx (sheet XML); the file is not
re-saved with a spreadsheet library, which would drop the cached results of every formula (Baseline, Proj <year>,
By_Recipient...) that announcement_model/workbook_inputs.py and allocation_model.py read. The workbook is flagged to
recalculate fully when Excel next opens it, so the summary sheets that add these shares up (By_Health_Area,
Health_Area_Summary, NonUS_by_Area, OECD_vs_IHME) show the new split; save it once in Excel to store them.

Before writing, the workbook's original (OECD Creditor Reporting System) HIV / TB / malaria split of Global Fund money
by country is saved to audit/logic_review/oecd_gf_shares_before_fix.csv (once), the evidence that the OECD split had
TB = 0 for combined TB/HIV grants and matched the Global Fund budgets where grants are separate.
Running it again changes nothing (the split is already in place).
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WB = ROOT / "OECD_aid_cut_projections.xlsx"
GF = ROOT / "audit" / "logic_review" / "gf_disease_split.csv"
BEFORE = ROOT / "audit" / "logic_review" / "oecd_gf_shares_before_fix.csv"
SHEET = "Base_Projection"
TRIPLES = {"base": ("Base share: HIV/AIDS", "Base share: Malaria", "Base share: Tuberculosis"),
           "projected": ("Projected share: HIV/AIDS", "Projected share: Malaria", "Projected share: Tuberculosis")}
KEYS = ("gf_hiv", "gf_malaria", "gf_tb")              # same order as the triples


def col_letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def sheet_path(z: zipfile.ZipFile) -> str:
    wbx = z.read("xl/workbook.xml").decode()
    rid = re.search(rf'<sheet[^>]*name="{SHEET}"[^>]*r:id="([^"]+)"', wbx).group(1)
    rels = z.read("xl/_rels/workbook.xml.rels").decode()
    tgt = re.search(rf'<Relationship[^>]*Id="{rid}"[^>]*Target="([^"]+)"', rels) or \
        re.search(rf'<Relationship[^>]*Target="([^"]+)"[^>]*Id="{rid}"', rels)
    t = tgt.group(1).lstrip("/")
    return t if t.startswith("xl/") else "xl/" + t


def main():
    bp = pd.read_excel(WB, SHEET)
    gf = pd.read_csv(GF).set_index("iso3")
    gf[list(KEYS)] = gf[list(KEYS)].div(gf[list(KEYS)].sum(axis=1), axis=0)
    bp["iso3"] = bp["Recipient ISO3"].replace({"XKV": "KSV"})          # the workbook's Kosovo code (as workbook_inputs)
    rows = bp[(bp["Channel name"] == "Global Fund") & bp["iso3"].isin(gf.index)]
    if not BEFORE.exists():                            # the workbook's own split, kept as evidence
        b = rows[rows.Baseline > 0]
        m = {k: (b.Baseline * b[c]).groupby(b["iso3"]).sum() for k, c in zip(("hiv", "malaria", "tb"), TRIPLES["base"])}
        t = pd.DataFrame(m)
        t = t.div(t.sum(axis=1), axis=0).add_prefix("oecd_crs_")
        t["baseline_kusd"] = b.groupby("iso3").Baseline.sum()
        t.index.name = "iso3"
        t.reset_index().round(6).to_csv(BEFORE, index=False)
    cols = [c for c in bp.columns if c != "iso3"]
    new = {}                                           # "AE3170" -> value
    for i, x in rows.iterrows():
        r = i + 2                                      # header is row 1
        for trip in TRIPLES.values():
            tot = sum(float(x[c]) for c in trip)
            for c, k in zip(trip, KEYS):
                new[f"{col_letter(cols.index(c))}{r}"] = round(tot * float(gf.at[x["iso3"], k]), 6)
    z = zipfile.ZipFile(WB)
    sp = sheet_path(z)
    xml = z.read(sp).decode("utf-8")
    hit = 0

    def sub(m):
        nonlocal hit
        ref = m.group(2)
        if ref not in new:
            return m.group(0)
        hit += 1
        attrs = re.sub(r'\st="[^"]*"', "", m.group(1))          # a typed number, never a string or formula
        return f'<c{attrs}><v>{new[ref]:g}</v></c>' if "<f" not in m.group(3) else m.group(0)

    xml2 = re.sub(r'<c((?:\s[^>]*?)?\sr="([A-Z]+\d+)"[^>]*?)(?:/>|>(.*?)</c>)', lambda m: sub(m) if m.group(3) is not None or m.group(0).endswith("/>") else m.group(0), xml, flags=re.S)
    missing = set(new) - {m for m in re.findall(r'<c[^>]*\sr="([A-Z]+\d+)"', xml) if m in new}
    assert not missing, f"cells not found in the sheet XML: {sorted(missing)[:10]}"
    wbx = z.read("xl/workbook.xml").decode()
    if "<calcPr" in wbx:
        wbx2 = re.sub(r"<calcPr([^>]*?)(\sfullCalcOnLoad=\"[^\"]*\")?(\s*/?)>", r'<calcPr\1 fullCalcOnLoad="1"\3>', wbx, count=1)
    else:
        wbx2 = wbx.replace("</workbook>", '<calcPr fullCalcOnLoad="1"/></workbook>')
    tmp = WB.with_suffix(".tmp.xlsx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for item in z.infolist():
            data = z.read(item.filename)
            if item.filename == sp:
                data = xml2.encode("utf-8")
            elif item.filename == "xl/workbook.xml":
                data = wbx2.encode("utf-8")
            out.writestr(item, data)
    z.close()
    shutil.move(tmp, WB)
    print(f"{SHEET}: {len(rows)} Global Fund rows, {hit} cells written ({len(new)} intended); "
          f"workbook set to recalculate on open")


if __name__ == "__main__":
    main()
