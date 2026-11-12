"""
Reallocation: the team's allocation model. The same post-cut health aid pool, re-divided so countries that can raise
more money at home lose more aid, while countries where aid saves lives cheaply and high-burden countries are
protected; four scenarios run through this dashboard's engine.

Data: model_data/allocation/*.csv, written by allocation_model.py from "ALLOCATION MODEL/Allocation_model_CostPerLife_
Need.xlsx" (the method is in that folder's two PDFs). Nothing here runs the engine: rebuild with python allocation_model.py.

Page order: the headline and rule picker; how the plan works; what it achieves (the four scenarios); who gains and
loses aid; who carries the funding gap; one country; services; other rules; caveats and sources. Detail that most
readers can skip sits in expanders.
"""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import allocation_model as am
import model_section as ms
import theme as th
from country_lists import ALLOWED_COUNTRIES
from model_section import chart, money, num, pill_row, rng

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "ALLOCATION MODEL"
e = html.escape

# scenario key -> (name, short name for chart axes, color). Order: the story (reallocation first, then new money)
SCEN = {"red": ("Current Split, No New Money", "Current Split,<br>No New Money", th.ROSE),
        "blue": ("Reallocated, No New Money", "Reallocated,<br>No New Money", th.BLUE),
        "yellow": ("Current Split + New Money", "Current Split<br>+ New Money", th.AMBER),
        "green": ("Reallocated + New Money", "Reallocated<br>+ New Money", th.TEAL)}
ORDER = list(SCEN)
RULE_PHRASE = {"cpl_need": "by ability to pay, cost per life saved and disease need",
               "cpl": "by ability to pay and cost per life saved",
               "need": "by ability to pay and disease need",
               "cap": "by ability to pay alone"}
PROTECTS = {"cpl_need": "countries where aid saves lives cheaply and countries with a high disease burden",
            "cpl": "countries where aid saves lives cheaply",
            "need": "countries with a high disease burden",
            "cap": None}
SERVICE_NAMES = {"hiv_art": ("HIV Treatment (ART)", "people on ART"), "hiv_pmtct": ("Preventing Mother-to-Child HIV Transmission", "women"),
                 "tb_ds": ("TB Treatment", "courses"), "tb_dr": ("Drug-Resistant TB Treatment", "courses"),
                 "mal_itn": ("Bednets", "people"), "mal_irs": ("Indoor Spraying", "people"),
                 "mal_cm": ("Malaria Testing & Treatment", "cases treated"), "imm": ("Routine Immunization", "children")}
TIER_COLORS = {"High": "#3D3747", "Medium": "#857E92", "Low": "#BDB6C8"}     # neutral: purple is TB and the brand
GAIN, LOSE = th.BLUE, th.ROSE
SOURCE_NOTE = "Team allocation model; this dashboard's model."
SHORT_RULE = {"cpl_need": "+ Cost per life + need (team plan)", "cpl": "+ Cost per life", "need": "+ Need",
              "cap": "Ability to pay only"}
N_MOVERS = 8                       # countries shown at each end of the "who gains and loses" chart


@st.cache_data(show_spinner=False)
def _load():
    return am.load()


def dot(color: str) -> str:
    return f"<span class='dot' style='background:{color}'></span>"


def pct(x: float, d: int = 0) -> str:
    return f"{x:.{d}%}"


def m_usd(x_m: float) -> str:
    """US$ millions -> the dashboard's money format."""
    return money(x_m * 1e6)


def signed_m(x_m: float) -> str:
    return ("+" if x_m >= 0 else "−") + m_usd(abs(x_m))


def bn(x_m: float) -> str:
    """US$ millions -> '$25.9B' (the short style for running text and cards)."""
    if abs(x_m) < 0.5:
        return "$0"
    return f"${x_m / 1e3:,.1f}B" if abs(x_m) >= 1e3 else f"${x_m:,.0f}M"


def lower_dis(b: str) -> str:
    return b if b in ("HIV", "TB") else b.lower()


def html_block(s: str):
    st.markdown(s.replace("$", "&#36;"), unsafe_allow_html=True)


def kicker(text: str, top: str = "1.6rem"):
    st.markdown(f"<div class='ed-kicker' style='margin:{top} 0 0.5rem'>{e(text)}</div>", unsafe_allow_html=True)


R = _load()
C = R["countries"]
RULES = R["rules"]
N_ALL, N_MATCHED = len(C), int(C["in_engine"].sum())
rule_keys = list(am.RULES)

# --------------------------------------------------------------------------- #
# Headline (drawn after the rule picker below it is read)
# --------------------------------------------------------------------------- #
lede_box = st.container()
with st.container(key="alloc_controls", border=True):
    c1, _ = st.columns([2.4, 1])
    with c1:
        rule = st.selectbox("Allocation Rule", rule_keys, key="al_rule", format_func=lambda k: am.RULES[k][0],
                            help="How the cut is shared out. Every rule divides the same total cut; only who absorbs "
                                 "it changes. The team's plan uses all three factors. Every number on this page "
                                 "follows this choice.")
        st.markdown(f"<div style='font-size:13.5px;color:{th.MUTED};margin:-6px 0 8px;line-height:1.45'>"
                    + ("The team's plan · " if rule == am.PLAN else "") + "protection strength: need "
                    f"{RULES.at[rule, 'need_k']:g}, cost per life {RULES.at[rule, 'eff_k']:g} (0 = off) · share given "
                    f"up (λ): {RULES.at[rule, 'lambda']:.1%}</div>", unsafe_allow_html=True)

DD = am.deaths_by_disease(R, rule)
TOT = DD.loc["Total"]
RANGE = {s: am.deaths_range(R, rule, None, s) for s in ORDER}
SAVED = TOT["red"] - TOT["blue"]
SAVED_R = am.deaths_range(R, rule, "red", "blue")
cut_col, aid_col, eff_col = f"cut_{rule}", f"aid_2026_{rule}", f"eff_capacity_{rule}"
POOL = C["aid_2026_proj"].sum()
TOTAL_CUT = C["aid_2023"].sum() - POOL
CAPACITY = C["capacity"].sum()
ZERO = int(RULES.at[rule, "zero_aid"])
LAM = float(RULES.at[rule, "lambda"])
proj_cut = C["aid_2023"] - C["aid_2026_proj"]
fills = {"red": 0.0, "blue": 0.0, "yellow": np.minimum(C["capacity"], proj_cut).sum(),
         "green": np.minimum(C["capacity"], C[cut_col]).sum()}
nk, ek = RULES.at[rule, "need_k"], RULES.at[rule, "eff_k"]

with lede_box:
    head = (f"Dividing the same {ms.money_words(POOL * 1e6)} of health aid {RULE_PHRASE[rule]} would cut extra deaths "
            f"by {SAVED / TOT['red']:.0%}")
    dek = (f"Health aid to these {N_ALL} countries is set to fall from {bn(C['aid_2023'].sum())} in 2023 to "
           f"{bn(POOL)} in 2026. Keeping that smaller total, this plan asks countries that can raise more money at home to "
           f"give up more aid" + (f", and protects {PROTECTS[rule]}." if PROTECTS[rule] else "."))
    big = [(num(TOT["red"]), "Extra Deaths, Current Split", "2026-2030, no new money"),
           (num(TOT["blue"]), "Extra Deaths, Reallocated", "same aid, no new money"),
           (num(SAVED), "Lives Saved by Reallocation Alone", f"range {rng(*SAVED_R)}"),
           (str(ZERO), "Countries Moved to Zero Aid", f"of {N_ALL}; their own money could replace it")]
    html_block("<div class='ed-lede'><div class='ed-kicker'>Allocation Model · Who Should Absorb the Aid Cut?</div>"
               f"<div class='ed-headline'>{e(head)}</div><p class='ed-dek'>{e(dek)}</p></div>"
               "<div class='ed-bignums'>"
               + "".join(f"<div class='ed-bignum'><div class='v'>{e(v)}</div><div class='l'>{e(l)}</div>"
                         f"<div class='n'>{e(n)}</div></div>" for v, l, n in big) + "</div>")
    st.markdown(th.source_html(None, f"Extra deaths from HIV, TB, malaria and immunization in the {N_MATCHED} countries "
                                     f"this dashboard models, relative to 2021-2023 aid. Ranges: the {th.RANGE_LABEL} "
                                     "(2.5th-97.5th percentile of 400 draws)."), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 1. How the plan works
# --------------------------------------------------------------------------- #
th.section_header("lightbulb", "How the Plan Works", rule=False)
protect_v = {"cpl_need": "Need + Cost per Life", "cpl": "Cost per Life", "need": "Disease Need", "cap": "None"}[rule]
protect_t = (f"For {PROTECTS[rule]}, what they could raise counts for less, so they give up less aid."
             if PROTECTS[rule] else "This rule protects no one beyond ability to pay. Choose another rule above to "
                                    "add protection for need or cost per life.")
cards = [("One Pot of Aid", bn(POOL), "a year in 2026",
          f"That is {bn(TOTAL_CUT)} less than in 2023 (OECD projection). The plan keeps this total and only "
          "changes which countries absorb the cut."),
         ("Those Who Can Pay Give More", f"{LAM:.1%}",
          "of what each country could raise" if not (nk or ek) else "of what each could raise, after protection",
          f"New taxes and budget shifts could add {bn(CAPACITY)} a year for health across the {N_ALL} countries. "
          + ("Each gives up the same share of its own amount" if not (nk or ek) else
             "Each gives up the same share of its amount once protection is applied")
          + f", never more than all its aid; {ZERO} reach zero aid."),
         ("Protection" if PROTECTS[rule] else "No Extra Protection", protect_v, "", protect_t)]
html_block("<div class='ed-cards three'>" + "".join(
    f"<div class='ed-card' style='border-top-color:{th.BRAND}'><div class='t'>{e(t)}</div>"
    f"<div class='big' style='font-size:26px'>{e(v)}</div>" + (f"<div class='lbl'>{e(n)}</div>" if n else "")
    + f"<div class='why'>{e(w)}</div></div>" for t, v, n, w in cards) + "</div>")

st.markdown("")
with st.expander("The formula, step by step"):
    need_t = rf"\times \dfrac{{1}}{{1+{nk:g}\,\text{{Need}}_i}}" if nk else ""
    eff_t = rf"\times \dfrac{{1}}{{1+{ek:g}\,\text{{Efficiency}}_i}}" if ek else ""
    st.latex(rf"\begin{{aligned}}\text{{Cut}}_i &= \min\left(\text{{Aid}}^{{2023}}_i,\;\lambda \times "
             rf"\text{{Effective}}_i\right)\\[6pt] \text{{Effective}}_i &= \text{{Capacity}}_i {need_t} {eff_t}"
             rf"\end{{aligned}}")
    best = C.loc[C["in_dashboard"]].sort_values("cost_per_life").iloc[0]
    k_need = nk or RULES.at[am.PLAN, "need_k"]
    st.markdown(ms._esc(f"""
1. **Capacity: what a country could raise for health, US$ a year.** New tax revenue (the Benitez et al. ceiling,
   reached at the Akitoby et al. pace, averaged over 2027-2030) times health's current share of government spending,
   plus a budget shift toward health: up to the country's own best 4-year rise in that share, phased in over
   2027-2030 and scaled by its Kharas & Sarkar fiscal space score (no shift at 48.8 or below, the full shift at 63.9
   and above). Total: {bn(CAPACITY)} a year.
2. **Need discount = 1 / (1 + {k_need:g} × need).** Need is the team disease-burden model's index: 80% HIV, TB and
   malaria burden, 20% dependence on health aid. It runs from 0 to {C['need'].max():.2f}, so the neediest country's
   capacity counts at {1 / (1 + k_need * C['need'].max()):.0%}. Countries outside that model take the median of
   their need group (high or low) in the team's country list.{'' if nk else ' *Off under this rule.*'}
3. **Efficiency discount = 1 / (1 + {(ek or RULES.at[am.PLAN, 'eff_k']):g} × efficiency).** Lives saved per US$1M of
   aid lost in this dashboard (HIV, TB, malaria and immunization, weighted by the country's aid mix), divided by the
   highest ({best['country']}, {money(best['cost_per_life'])} per life saved), so efficiency runs from 0 to 1. The
   {int((~C['in_dashboard']).sum())} countries this dashboard does not model take their income group's
   average.{'' if ek else ' *Off under this rule.*'}
4. **λ = {LAM:.1%}.** One number for every country: the share of its effective capacity (capacity × both
   discounts) it gives up. It is set so the cuts add up to exactly {bn(TOTAL_CUT)} a year. A country whose cut would be
   more than all its aid goes to zero instead, and λ rises a little so the others cover the difference.
"""))

# --------------------------------------------------------------------------- #
# 2. What it achieves: the four scenarios
# --------------------------------------------------------------------------- #
th.section_header("grid_view", "What It Achieves")
st.markdown(ms._esc(f"Extra deaths over 2026-2030 in the {N_MATCHED} countries this dashboard models, for each mix "
                    f"of two choices:"))


def mcell(s: str) -> str:
    name, _, col = SCEN[s]
    lo, hi = RANGE[s]
    chg = "" if s == "red" else f"<small>−{1 - TOT[s] / TOT['red']:.0%}</small>"
    if s in ("red", "blue"):
        note = f"range {rng(lo, hi)}"
    else:
        gap_left = TOTAL_CUT - fills[s]
        note = (f"range {rng(lo, hi)} · new money fills {bn(fills[s])} of the {bn(TOTAL_CUT)} cut"
                if gap_left > 0.5 else f"new money fills the whole {bn(TOTAL_CUT)} cut")
    tip = f"{name}: {TOT[s]:,.0f} extra deaths 2026-2030 (range {lo:,.0f} to {hi:,.0f})."
    return (f"<div class='ed-mcell' style='border-left-color:{col}' title='{e(tip, quote=True)}'>"
            f"<div class='k' style='color:{col}'>{e(name)}</div><div class='v'>{num(TOT[s])}{chg}</div>"
            f"<div class='n'>{e(note)}</div></div>")


html_block("<div class='ed-matrix'>"
           "<div class='corner'></div><div class='hd'>No new money</div><div class='hd'>With new money</div>"
           "<div class='rh'>Current split<small>OECD-projected 2026 aid</small></div>" + mcell("red") + mcell("yellow")
           + "<div class='rh'>Reallocated<small>" + e(am.RULES[rule][0].lower().capitalize()) + "</small></div>"
           + mcell("blue") + mcell("green") + "</div>")

fig = go.Figure()
for b in am.BUCKETS:
    fig.add_bar(name=b, x=[SCEN[s][1] for s in ORDER], y=[DD.at[b, s] for s in ORDER],
                marker_color=th.DISEASE_COLORS[b], width=0.55,
                hovertemplate=f"<b>{b}</b><br>%{{y:,.0f}} extra deaths<extra></extra>")
for s in ORDER:
    lab = f"<b>{num(TOT[s])}</b>" + ("" if s == "red" else
                                      f"  <span style='color:{th.MUTED}'>−{1 - TOT[s] / TOT['red']:.0%}</span>")
    fig.add_annotation(x=SCEN[s][1], y=TOT[s], text=lab, showarrow=False, yshift=12, font=dict(size=13, color=th.INK))
fig.update_layout(barmode="stack", height=440, showlegend=False,
                  title=th.title_sub("Extra Deaths by Disease, 2026-2030",
                                     th.color_key((b, th.DISEASE_COLORS[b]) for b in am.BUCKETS)),
                  yaxis=dict(tickformat=",.0f", range=[0, TOT.max() * 1.13]))
chart(fig, source=SOURCE_NOTE, note="% = change from the current split with no new money.")
dis_down = [b for b in am.BUCKETS if DD.at[b, "blue"] < DD.at[b, "red"]]
dis_up = [b for b in am.BUCKETS if DD.at[b, "blue"] > DD.at[b, "red"]]
if dis_up:
    st.markdown(ms._esc(
        f"**The catch: {' and '.join(dis_up)} deaths rise.** Reallocation alone saves lives from "
        + (', '.join(lower_dis(b) for b in dis_down[:-1]) + ' and ' + lower_dis(dis_down[-1]) if len(dis_down) > 1
           else lower_dis(dis_down[0])) + ", but "
        + (f"TB deaths go from {DD.at['TB', 'red']:,.0f} to {DD.at['TB', 'blue']:,.0f}: aid moves away from large "
           f"middle-income countries with heavy TB burdens. " if "TB" in dis_up else "")
        + "Adding countries' own money closes the gap."))

with st.expander("How much of the gain comes from new money, and how much from reallocation?"):
    nm = ((TOT["red"] - TOT["yellow"]) + (TOT["blue"] - TOT["green"])) / 2
    ra = ((TOT["red"] - TOT["blue"]) + (TOT["yellow"] - TOT["green"])) / 2
    st.markdown(ms._esc(
        f"Going from the current split with no new money to reallocated aid plus new money saves about "
        f"{ms.round_words(TOT['red'] - TOT['green'])} lives. The two changes interact, so each one's share depends on "
        f"which comes first; averaging both orders gives **about {ms.round_words(nm)} to new money** and **about "
        f"{ms.round_words(ra)} to reallocation**. This is an "
        f"accounting split of these scenarios, not a causal estimate."))

# --------------------------------------------------------------------------- #
# 3. Who gains and who loses aid
# --------------------------------------------------------------------------- #
th.section_header("swap_vert", "Who Gains and Who Loses Aid",
                  help="Each country's 2026 aid under this plan compared with the OECD projection, US$ a year.")
dv = (C[aid_col] - C["aid_2026_proj"])
n_gain, n_lose = int(RULES.at[rule, "gain_vs_proj"]), int(RULES.at[rule, "lose_vs_proj"])
movers = pd.concat([dv.nlargest(N_MOVERS), dv.nsmallest(N_MOVERS).iloc[::-1]])
names = C.loc[movers.index, "country"].replace({"Democratic Republic of the Congo": "DR Congo"})
cdm = np.column_stack([C.loc[movers.index, "aid_2023"].map(m_usd), C.loc[movers.index, "aid_2026_proj"].map(m_usd),
                       C.loc[movers.index, aid_col].map(m_usd), C.loc[movers.index, "capacity"].map(m_usd)])
fig = go.Figure(go.Bar(
    y=names, x=movers, orientation="h", width=0.68, marker_line_width=0,
    marker_color=[GAIN if v >= 0 else LOSE for v in movers], customdata=cdm,
    text=[("+" if v >= 0 else "−") + f"${abs(v):,.0f}M" for v in movers], textposition=["inside" if abs(v) >= 0.6 * movers.abs().max() else "outside" for v in movers],
    cliponaxis=False,
    insidetextanchor="end", insidetextfont=dict(size=12, color=th.SURFACE), outsidetextfont=dict(size=12, color=th.INK),
    hovertemplate=("<b>%{y}</b><br>%{x:+$,.1f}M a year vs the OECD projection<br>2023 aid %{customdata[0]} · OECD 2026 "
                   "%{customdata[1]} · this plan %{customdata[2]}<br>Could raise %{customdata[3]} a year for health"
                   "<extra></extra>")))
lo_x, hi_x = float(min(movers.min(), 0)), float(max(movers.max(), 0))
pad = (hi_x - lo_x) * 0.2                          # room for the value labels outside the bars
fig.update_layout(height=40 + 26 * len(movers) + 80, showlegend=False, margin=dict(t=80, l=10, r=10),
                  uniformtext=dict(minsize=12, mode="show"),
                  title=th.title_sub(f"The {N_MOVERS} Biggest Gains and Losses",
                                     "This plan minus the OECD projection, US$ a year"),
                  xaxis=dict(range=[lo_x - pad, hi_x + pad], tickformat="$,.0f", ticksuffix="M", nticks=7),
                  yaxis=dict(autorange="reversed", ticklabelstandoff=8))
chart(fig, source=SOURCE_NOTE, note="Hover for each country's aid and what it could raise at home.")
loser = C.loc[dv.idxmin()]
st.markdown(ms._esc(
    f"Compared with the OECD projection, **{n_gain} countries get more aid and {n_lose} get less**; {ZERO} get none. "
    f"The biggest loser is {loser['country']}: its aid falls from {bn(loser['aid_2023'])} in 2023 to "
    f"{bn(loser[aid_col])} (the OECD projected {bn(loser['aid_2026_proj'])}), but it could raise "
    f"{bn(loser['capacity'])} a year for health at home."))

with st.expander(f"Every country's aid, split four ways (all {N_ALL})"):
    A = C.copy()
    A["model"], A["proj"] = A[aid_col], A["aid_2026_proj"]
    A["keep"] = np.minimum(A["model"], A["proj"])
    A["gain"] = (A["model"] - A["proj"]).clip(lower=0)
    A["away"] = (A["proj"] - A["model"]).clip(lower=0)
    A["rest"] = A["aid_2023"] - np.maximum(A["model"], A["proj"])
    A = A.assign(rank=A[eff_col] / A["aid_2023"]).sort_values("rank")
    parts = [("keep", "Aid the country keeps", th.blend(th.BLUE, 0.38)),
             ("gain", "Aid gained in the reallocation", th.BLUE),
             ("away", "Aid moved away, replaced by new money", th.TEAL),
             ("rest", "Rest of the cut, replaced by new money", th.blend(th.TEAL, 0.4))]
    fig = go.Figure()
    cd = np.stack([A["country"], A["aid_2023"].map(m_usd), A["proj"].map(m_usd), A["model"].map(m_usd)], axis=-1)
    for k, lbl, col in parts:
        fig.add_bar(name=lbl, x=A["country"], y=A[k] / A["aid_2023"] * 100, marker_color=col, marker_line_width=0,
                    customdata=np.column_stack([cd, (A[k]).map(m_usd)]),
                    hovertemplate=("<b>%{customdata[0]}</b><br>" + lbl + ": %{y:.1f}% (%{customdata[4]})<br>"
                                   "2023 aid %{customdata[1]} · OECD 2026 %{customdata[2]} · this plan %{customdata[3]}"
                                   "<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.12, height=470, margin=dict(t=100),
                      title=th.title_sub("Each Country's 2023 Aid, With New Money Filling the Cut",
                                         "One bar per country, ranked by how much it could raise relative to its aid"),
                      xaxis=dict(showticklabels=False, title="← Least able to replace aid · most able →"),
                      yaxis=dict(range=[0, 104], ticksuffix="%", title="Share of 2023 health aid"))
    chart(fig, source=SOURCE_NOTE,
          note="Reallocated + new money. Every country's cut is smaller than what it could raise, so new money covers "
               "all of it in this scenario. Hover for each country.")

# --------------------------------------------------------------------------- #
# 4. Who carries the funding gap
# --------------------------------------------------------------------------- #
th.section_header("money_off", "Who Carries the Funding Gap",
                  help="Funding gap: aid lost since 2023 that new domestic money does not replace, US$ a year, all "
                       f"{N_ALL} countries.")
gap = {"red": proj_cut, "blue": C[cut_col], "yellow": (proj_cut - C["capacity"]).clip(lower=0),
       "green": (C[cut_col] - C["capacity"]).clip(lower=0)}
tiers = C["need_tier"]
fig = go.Figure()
for t in ["High", "Medium", "Low"]:
    fig.add_bar(name=f"{t} need", x=[SCEN[s][1] for s in ORDER],
                y=[gap[s][tiers == t].sum() / 1e3 for s in ORDER], marker_color=TIER_COLORS[t], width=0.55,
                hovertemplate=f"<b>{t} need</b><br>%{{y:$,.2f}}B a year<extra></extra>")
for s in ORDER:
    fig.add_annotation(x=SCEN[s][1], y=gap[s].sum() / 1e3, text=f"<b>{bn(gap[s].sum())}</b>", showarrow=False,
                       yshift=12, font=dict(size=13, color=th.INK))
fig.update_layout(barmode="stack", height=460, showlegend=False, margin=dict(t=100),
                  title=th.title_sub("Funding Gap Left Each Year, by Disease Need",
                                     th.color_key((f"{t} ({int((tiers == t).sum())})", TIER_COLORS[t])
                                                  for t in ["High", "Medium", "Low"])),
                  yaxis=dict(tickformat="$,.0f", ticksuffix="B", range=[0, TOTAL_CUT / 1e3 * 1.15]))
chart(fig, source=SOURCE_NOTE,
      note=f"US$ a year, all {N_ALL} countries. Need groups (number of countries): thirds by the need index; countries "
           "tied at a fallback value share a third, so the groups are not equal.")
hi_r, hi_b = gap["red"][tiers == "High"].sum(), gap["blue"][tiers == "High"].sum()
lo_r, lo_b = gap["red"][tiers == "Low"].sum(), gap["blue"][tiers == "Low"].sum()
st.markdown(ms._esc(f"With no new money the total gap stays at {bn(TOTAL_CUT)} a year; reallocation only moves it. "
                    f"The high-need countries' share falls from {bn(hi_r)} to {bn(hi_b)}, and the low-need "
                    f"countries' rises from {bn(lo_r)} to {bn(lo_b)}."))

# --------------------------------------------------------------------------- #
# 5. One country
# --------------------------------------------------------------------------- #
th.section_header("public", "Look Up a Country",
                  help="Aid figures are all health aid (OECD), US$ a year. Deaths cover HIV, TB, malaria and "
                       "immunization in the countries this dashboard models.")
CR = am.country_results(R, rule)
names_sorted = C["country"].sort_values()
cur = st.session_state.get("sel_iso3", "NGA")
if st.session_state.get("al_country") not in C.index:
    st.session_state["al_country"] = cur if cur in C.index else "NGA"
c1, _ = st.columns([1.3, 2])
with c1:
    iso = st.selectbox("Country", list(names_sorted.index), key="al_country", format_func=lambda i: C.at[i, "country"])
if iso in ALLOWED_COUNTRIES:
    st.session_state["sel_iso3"] = iso                 # the other pages follow (only countries they cover)
r = C.loc[iso]
cut = r[cut_col]
capped = cut >= r["aid_2023"] - 1e-6
ncut, cutp = r["aid_2023"] - r["aid_2026_proj"], cut / r["aid_2023"] if r["aid_2023"] else 0.0
nd, ed = r[f"need_disc_{rule}"], r[f"eff_disc_{rule}"]
pill_row([("2023 Aid", m_usd(r["aid_2023"]), "a year, all health aid"),
          ("OECD 2026 Projection", m_usd(r["aid_2026_proj"]), f"cut {pct(ncut / r['aid_2023'], 1)}"),
          ("This Plan, 2026", m_usd(r[aid_col]), f"cut {pct(cutp, 1)}"),
          ("Change vs Projection", signed_m(r[aid_col] - r["aid_2026_proj"]), "a year")], 4)
cap_tip = (f"New tax revenue {m_usd(r['new_tax'])} × health share {r['health_share']:.1%} = "
           f"{m_usd(r['tax_for_health'])}; budget shift {m_usd(r['budget_shift'])} (fiscal space room {r['room']:.0%} × "
           f"best 4-year rise {r['shift_cap']:.2%} ({r['shift_cap_source']}) × government spending × 0.625 phase-in)."
           if r["capacity"] > 0 else "No new tax revenue or budget shift is counted (missing data or already at tax "
                                     "potential).")
cpl_txt = (f"{money(r['cost_per_life'])} per life saved in this dashboard" if r["in_dashboard"]
           else f"Not modelled by this dashboard: the {r['income'].lower()} average")
pill_row([("Could Raise for Health", m_usd(r["capacity"]), "a year", cap_tip),
          ("Disease Need", f"{r['need']:.2f}", f"discount × {nd:.2f}" if nk else "not used by this rule",
           r["need_source"]),
          ("Cost-per-Life Score", f"{r['efficiency']:.2f}", f"discount × {ed:.2f}" if ek else "not used by this rule",
           "Efficiency, 0-1: lives saved per US$1M of aid relative to the best country. " + cpl_txt),
          ("After Protection", m_usd(r[eff_col]), "cut = all of its 2023 aid" if capped else f"× λ = cut {m_usd(cut)}",
           "What it could raise × the need and cost-per-life discounts (its effective capacity). The cut is λ times "
           "this, but never more than all of the country's 2023 aid.")], 4)
_n_disc = int(bool(nk)) + int(bool(ek))
st.markdown(ms._esc(
    f"**How {r['country']}'s cut is set.** {r['country']} has no data on health's share of government spending, so it "
    f"is counted as able to raise nothing, and its aid is not cut." if r["capacity"] <= 0 else
    f"**How {r['country']}'s cut is set.** It could raise {m_usd(r['capacity'])} a year for health"
    + (f"; after the discount{'s' if _n_disc > 1 else ''} ({'× ' + format(nd, '.2f') + ' for need' if nk else ''}"
       f"{', ' if nk and ek else ''}{'× ' + format(ed, '.2f') + ' for cost per life' if ek else ''}) that counts as "
       f"{m_usd(r[eff_col])}" if (nk or ek) else "")
    + (f". Its cut is λ = {LAM:.1%} of that: {m_usd(cut)}, or {pct(cutp, 1)} of its 2023 aid."
       if not capped else
       f". λ = {LAM:.1%} of that would be more than all of its {m_usd(r['aid_2023'])} of 2023 aid, so its aid goes to "
       f"zero and its own money is expected to replace it.")))
if iso in CR.index:
    cr = CR.loc[iso]
    rise = cr["deaths_blue"] - cr["deaths_red"]
    st.markdown(ms._esc(
        (f"**Deaths rise here under this plan**, by {rise:,.0f}, because {r['country']} gives up more aid than in the "
         f"OECD projection. " if rise >= 0.5 and r[aid_col] < r["aid_2026_proj"] else "")
        + f"Extra deaths 2026-2030 in {r['country']}: **{cr['deaths_red']:,.0f}** with the current split and "
        f"**{cr['deaths_blue']:,.0f}** with reallocated aid, both with no new money; "
        + ("none" if max(cr["deaths_yellow"], cr["deaths_green"]) < 0.5 else
           f"{cr['deaths_yellow']:,.0f} and {cr['deaths_green']:,.0f}")
        + " once new money is added."))
else:
    st.caption(f"{r['country']} is not one of the countries this dashboard models, so no deaths are estimated.")

with st.expander(f"All {N_ALL} countries in one table"):
    T = pd.DataFrame({
        "Country": C["country"], "Income Group": C["income"],
        "2023 Aid ($M)": C["aid_2023"], "OECD 2026 ($M)": C["aid_2026_proj"], "This Plan 2026 ($M)": C[aid_col],
        "Change vs OECD ($M)": C[aid_col] - C["aid_2026_proj"], "Could Raise ($M/yr)": C["capacity"],
        "Need": C["need"], "Cost per Life Saved ($)": C["cost_per_life"], "Effective Capacity ($M)": C[eff_col],
        "Deaths, Current Split": CR["deaths_red"].reindex(C.index),
        "Deaths, Reallocated": CR["deaths_blue"].reindex(C.index)})
    T["Lives Saved"] = T["Deaths, Current Split"] - T["Deaths, Reallocated"]
    T = T.sort_values("Change vs OECD ($M)", ascending=False).round(
        {c: 1 for c in T.columns if "($M" in c} | {"Need": 2, "Cost per Life Saved ($)": 0, "Deaths, Current Split": 0,
                                                     "Deaths, Reallocated": 0, "Lives Saved": 0})
    fm = st.column_config.NumberColumn
    st.dataframe(T, hide_index=True, height=420, **ms.WIDE, column_config={
        "Country": st.column_config.TextColumn(pinned=True, width="medium"),
        "2023 Aid ($M)": fm(format="localized"), "OECD 2026 ($M)": fm(format="localized"),
        "This Plan 2026 ($M)": fm(format="localized"), "Change vs OECD ($M)": fm(format="%+.1f"),
        "Could Raise ($M/yr)": fm(format="localized"), "Need": fm(format="%.2f"),
        "Cost per Life Saved ($)": fm(format="localized", help="Blank: not modelled by this dashboard (uses its "
                                                                "income-group average efficiency)"),
        "Effective Capacity ($M)": fm(format="localized"), "Deaths, Current Split": fm(format="localized"),
        "Deaths, Reallocated": fm(format="localized"), "Lives Saved": fm(format="localized")})
    st.markdown(th.source_html(SOURCE_NOTE, "Deaths: no new money, 2026-2030; blank where the dashboard does not "
                                            "model the country. Sorted by change versus the OECD projection."),
                unsafe_allow_html=True)
    st.download_button("Download This Table (CSV)", T.to_csv(index=False).encode(),
                       file_name=f"allocation_{rule}.csv", mime="text/csv", key="dl_alloc_table")

# --------------------------------------------------------------------------- #
# 6. Services and coverage
# --------------------------------------------------------------------------- #
th.section_header("medical_services", "Services and Coverage",
                  help=f"Coverage across the {N_MATCHED} modelled countries, weighted by each country's population in "
                       "need of the service. Reallocated + new money keeps every service at its 2021-2023 coverage.")
COV, UN = am.coverage(R, rule), am.units_lost(R, rule)
head_cells = "".join(f"<th>{dot(SCEN[s][2])}{SCEN[s][1]}</th>" for s in ORDER)
body = []
for ln in am.SERVICES:
    nm_, unit = SERVICE_NAMES[ln]
    cells = []
    for s in ORDER:
        lost = UN.at[ln, s]
        cells.append(f"<td><span class='big'>{COV.at[ln, s]:.1%}</span><br>"
                     f"<span class='nm'>{num(lost) + ' lost' if lost >= 0.5 else 'none lost'}</span></td>")
    body.append(f"<tr><td class='svc'><b>{e(nm_)}</b><br><span class='sub'>units: {e(unit)} a year</span></td>"
                + "".join(cells) + "</tr>")
tot_cells = "".join(f"<td><b>{num(UN[s].sum())}</b></td>" for s in ORDER)
body.append(f"<tr class='tot'><td class='svc'><b>Service Units Lost a Year</b><br><span class='sub'>not unique "
            f"people: units differ and overlap</span></td>{tot_cells}</tr>")
html_block("<div class='ed-nyt-wrap'><table class='ed-nyt sticky' style='min-width:720px'><colgroup><col style='width:28%'>"
           + "<col style='width:18%'>" * 4 + f"</colgroup><thead><tr><th>Service</th>{head_cells}</tr></thead><tbody>"
           + "".join(body) + "</tbody></table></div>")
st.markdown(th.source_html(SOURCE_NOTE, "Coverage now minus the modelled drop, average 2026-2030; units lost a year "
                                        "relative to 2021-2023 coverage. The model counts losses only: money above "
                                        "2023 aid buys no extra coverage here."), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 7. Other ways to divide the cut
# --------------------------------------------------------------------------- #
th.section_header("compare_arrows", "Other Ways to Divide the Cut",
                  help="Same aid pool, same new money, same model: only the rule that divides the cut changes. With "
                       "new money every rule closes every gap, so the rules differ only without it.")
rows = []
cap_blue = am.deaths_by_disease(R, "cap").at["Total", "blue"]
for k in rule_keys:
    d = am.deaths_by_disease(R, k).loc["Total"]
    rows.append({"key": k, "label": am.RULES[k][0], "deaths": d["blue"], "saved": d["red"] - d["blue"],
                 "range": am.deaths_range(R, k, "red", "blue"), "zero": int(RULES.at[k, "zero_aid"]),
                 "lam": float(RULES.at[k, "lambda"]), "vs_cap": cap_blue - d["blue"],
                 "vs_cap_r": am.rule_gain_range(R, "cap", k) if k != "cap" else None})
RR = pd.DataFrame(rows).iloc[::-1]
fig = go.Figure(go.Bar(
    y=[f"{SHORT_RULE[k]}<br><b>{num(v)}</b>" for k, v in zip(RR["key"], RR["saved"])], x=RR["saved"],
    orientation="h",
    width=0.6, marker_line_width=0,
    marker_color=[th.BLUE if k == rule else th.blend(th.BLUE, 0.35) for k in RR["key"]],
    error_x=dict(type="data", symmetric=False, color=th.MUTED, thickness=1.3, width=6,
                 array=[r_[1] - s_ for r_, s_ in zip(RR["range"], RR["saved"])],
                 arrayminus=[s_ - r_[0] for r_, s_ in zip(RR["range"], RR["saved"])]),
    customdata=np.column_stack([RR["range"].map(lambda t: rng(*t)), RR["zero"], RR["lam"].map(lambda x: f"{x:.1%}"),
                                RR["label"]]),
    hovertemplate=("<b>%{customdata[3]}</b><br>%{x:,.0f} lives saved vs the current split<br>range %{customdata[0]}<br>"
                   "λ %{customdata[2]} · %{customdata[1]} countries at zero aid<extra></extra>")))
fig.update_layout(height=360, showlegend=False, margin=dict(t=80, r=20),
                  title=th.title_sub("Lives Saved by Reallocation Alone, by Rule",
                                     "2026-2030, compared with the current split, no new money"),
                  xaxis=dict(tickformat="~s", nticks=5, range=[0, max(r_[1] for r_ in RR["range"]) * 1.08]),
                  yaxis=dict(ticklabelstandoff=10))
chart(fig, source=SOURCE_NOTE, note=f"Every rule starts from ability to pay. Whiskers: the {th.RANGE_LABEL}. The "
                                     "selected rule is dark blue.")
best_rule = RR.sort_values("saved").iloc[-1]
if best_rule["key"] != "cap":
    st.markdown(ms._esc(
        f"**{best_rule['label']} saves the most lives here**, {best_rule['vs_cap']:,.0f} more than ability to pay "
        f"alone. Each protection shields some countries, so the cut falls on fewer others and more of them end at zero "
        f"aid ({int(RR.set_index('key').at['cap', 'zero'])} with ability to pay alone, "
        f"{int(RR.set_index('key').at[am.PLAN, 'zero'])} with the team plan)."))

with st.expander("Why cost per life favours countries with large TB programs"):
    MED = am.disease_medians(R)
    pill_row([(b, f"${MED.at[b, 'cost_per_life']:,.0f}", "per life saved",
               f"{MED.at[b, 'lives_per_musd']:,.0f} lives saved per US$1M of aid lost. Median over the "
               f"{int(MED.at[b, 'countries'])} countries, of all 70 in the dashboard, where {b} is at least 5% of their HIV, "
               "TB, malaria and immunization aid.") for b in am.BUCKETS], 4)
    st.markdown(ms._esc(
        "In this model, TB treatment is the cheapest way to save a life (typical cost per life saved above, median "
        "over the dashboard's countries). A country's efficiency weights these costs by its own aid mix, so countries "
        "with large TB programs score as efficient and are protected. That is why the rules using cost per life save "
        "the most lives here, and why TB deaths rise least under them."))
    tbl = "".join(
        f"<tr{' style=\"background:' + th.SURFACE_TINT + '\"' if q['key'] == rule else ''}><td class='svc'><b>"
        f"{e(q['label'])}</b>{' <span class=\"sub\">(team plan)</span>' if q['key'] == am.PLAN else ''}</td>"
        f"<td>{q['lam']:.1%}</td><td>{q['zero']}</td><td>{num(q['deaths'])}</td>"
        f"<td>{'—' if q['vs_cap_r'] is None else '+' + num(q['vs_cap']) + '<br><span class=nm>' + rng(*q['vs_cap_r']) + '</span>'}"
        f"</td></tr>" for _, q in RR.iloc[::-1].iterrows())
    html_block("<div class='ed-nyt-wrap' style='margin-top:1rem'><table class='ed-nyt' style='min-width:640px'>"
               "<colgroup><col style='width:36%'>" + "<col style='width:16%'>" * 4 + "</colgroup><thead><tr>"
               "<th>Rule</th><th>λ</th><th>Countries at Zero Aid</th><th>Extra Deaths, No New Money</th>"
               "<th>Lives Saved vs Ability to Pay Only</th></tr></thead><tbody>" + tbl + "</tbody></table></div>")

# --------------------------------------------------------------------------- #
# 8. Caveats and sources
# --------------------------------------------------------------------------- #
th.section_header("report", "Reading These Results")
# the team's results PDF (11 November 2026): extra deaths 2026-2030, current split / + new money / team plan reallocated
PDF_DEATHS = {"red": 898_897, "yellow": 199_893, "blue": 506_894}
_now = am.deaths_by_disease(R, am.PLAN).loc["Total"]
_gap = max(abs(_now[k] / v - 1) for k, v in PDF_DEATHS.items())
_same = all(round(_now[k]) == v for k, v in PDF_DEATHS.items())       # the PDF prints whole deaths
PDF_NOTE = ("These results match the team's results PDF exactly." if _same else
            f"This page uses the dashboard's current model, which has changed since the team's results PDF: total "
            f"deaths differ from the PDF by up to {_gap:.1%}, some service counts (such as indoor spraying) by more.")
st.markdown(ms._esc(f"""
- **Built to score well here.** Cost per life comes from the same model that scores the results, so rules that use it
  are designed to look good by this measure. Treat their lives saved as an upper bound; ability to pay + need is the
  more independent comparison.
- **Losses only.** The model counts what is lost when aid falls below its 2021-2023 level. The
  {bn(CAPACITY - fills['green'])} a year of new money beyond what fills the cut buys nothing extra on this page.
- **Two country sets.** The allocation covers {N_ALL} countries and all health aid; deaths and coverage cover the
  {N_MATCHED} of them this dashboard models, for HIV, TB, malaria and immunization only.
- **What countries could raise is not what they will.** The Kharas & Sarkar index measures room for new spending, not
  budget flexibility; using it shrinks the budget shift, so the plan assumes less new money. Kosovo has no
  health-share data, so it could raise nothing and is never cut; Azerbaijan and Somalia have no fiscal space score,
  so only their tax revenue counts.
- **The strength of the protection (k = 4) is a choice.** Where cost per life is used, the most efficient country
  counts at a fifth of what it could raise; where need is used, need tops out at {C['need'].max():.2f}, so the
  neediest country counts at about a third.
- **Fixed scenarios, not forecasts.** Each scenario holds one set of choices for 2026-2030; none is a prediction.
- **{'Same as the team PDF' if _same else 'Updated since the team PDF'}.** {PDF_NOTE}
"""))
kicker("Source files", "1.4rem")
cols = st.columns(3)
for col, (fname, label, mime) in zip(cols, [
        ("Allocation_model_CostPerLife_Need.xlsx", "Workbook (Excel)",
         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("Allocation_Model_Explanation.pdf", "Method (PDF)", "application/pdf"),
        ("Allocation Results.pdf", "Results (PDF)", "application/pdf")]):
    p = SRC / fname
    if p.exists():
        with col:
            st.download_button(label, p.read_bytes(), file_name=fname, mime=mime, key=f"dl_alloc_{fname}", **ms.WIDE)
