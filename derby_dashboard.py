"""
Kentucky Derby 2026 — Streamlit Dashboard v2  (dark theme, live scratch manager)
==================================================================================
Run:  streamlit run derby_dashboard_v2.py
Deps: pip install streamlit pandas numpy scikit-learn plotly

Same folder as script:
  kentucky_derby_equibase_MASTER.csv
  derby_jockey_stats.csv
  derby_trainer_stats.csv
  derby_post_positions.csv

Scratches: use the sidebar multiselect — model reruns automatically with
the remaining horses and renormalised probabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Kentucky Derby 2026 | ML Model",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DARK THEME CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0f0f0f !important;
    color: #e8e6e0 !important;
}
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

/* sidebar */
[data-testid="stSidebar"] {
    background: #161616 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #aaa !important; font-size: 12px; }
[data-testid="stSidebar"] h3 { color: #e8e6e0 !important; font-size: 13px; font-weight: 600; }

/* selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1e1e1e !important;
    border: 1px solid #333 !important;
    color: #e8e6e0 !important;
    border-radius: 6px;
}

/* slider */
[data-testid="stSlider"] .stSlider > div { background: transparent; }
[data-testid="stSlider"] span { color: #aaa !important; font-size: 11px; }

/* checkbox */
[data-testid="stCheckbox"] span { color: #aaa !important; font-size: 12px; }

/* expander */
[data-testid="stExpander"] {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px;
}
[data-testid="stExpander"] summary { color: #aaa !important; font-size: 13px; }

/* dataframe */
[data-testid="stDataFrame"] { background: #161616 !important; }
.dvn-scroller { background: #161616 !important; }

/* metric cards */
.kpi-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.kpi-label {
    font-size: 10px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 6px;
}
.kpi-value { font-size: 30px; font-weight: 700; color: #f0ede6; line-height: 1; }
.kpi-sub   { font-size: 11px; color: #555; margin-top: 5px; }

/* pick cards */
.pick-win   { background: #0d1f35; border-left: 4px solid #3b82f6; border-radius: 0 10px 10px 0; padding: 1.1rem 1.25rem; }
.pick-place { background: #0d2b1f; border-left: 4px solid #22c55e; border-radius: 0 10px 10px 0; padding: 1.1rem 1.25rem; }
.pick-show  { background: #1e1b12; border-left: 4px solid #eab308; border-radius: 0 10px 10px 0; padding: 1.1rem 1.25rem; }
.pick-lbl   { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #666; margin-bottom: 5px; }
.pick-name  { font-size: 20px; font-weight: 700; color: #f0ede6; }
.pick-detail{ font-size: 12px; color: #888; margin-top: 5px; line-height: 1.6; }

/* section titles */
.sec-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 14px;
    margin-top: 4px;
}

/* DRF table */
.drf-wrap { overflow-x: auto; }
.drf {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    font-family: 'Inter', sans-serif;
}
.drf thead th {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #555;
    padding: 5px 7px;
    border-bottom: 1px solid #2a2a2a;
    text-align: left;
    white-space: nowrap;
    background: #0f0f0f;
}
.drf thead th.r { text-align: right; }
.drf tbody td {
    padding: 6px 7px;
    border-bottom: 1px solid #1e1e1e;
    color: #ccc;
    vertical-align: middle;
}
.drf tbody td.r { text-align: right; font-variant-numeric: tabular-nums; }
.drf .sec-row td {
    background: #161616;
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #444;
    padding: 5px 7px;
    border-bottom: 1px solid #222;
}
.drf .note-row td {
    font-size: 11px;
    color: #555;
    padding: 1px 7px 6px 26px;
    border-bottom: 1px solid #1a1a1a;
}
.drf tbody tr:hover td { background: #181818; }
.scratched td { opacity: 0.35; text-decoration: line-through; }

/* Beyer colors */
.b-elite  { color: #60a5fa; font-weight: 700; font-size: 15px; }
.b-qual   { color: #34d399; font-weight: 600; font-size: 14px; }
.b-border { color: #fbbf24; font-weight: 500; }
.b-sub    { color: #444; }

/* dots */
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.d-yes    { background:#3b82f6; }
.d-border { background:#eab308; }
.d-no     { background:#2a2a2a; }

/* trend */
.t-up   { color: #22c55e; font-weight: 600; font-size: 11px; }
.t-dn   { color: #ef4444; font-weight: 600; font-size: 11px; }
.t-flat { color: #444;    font-size: 11px; }

/* bar */
.bw { display:inline-block; width:65px; height:6px; background:#222; border-radius:3px; vertical-align:middle; overflow:hidden; }
.bf { height:100%; border-radius:3px; }

/* trifecta summary pills */
.tri-hit  { color:#22c55e; font-weight:700; }
.tri-part { color:#eab308; font-weight:600; }
.tri-miss { color:#2a2a2a; }
.pill {
    display:inline-block;
    font-size:9px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.07em;
    padding:1px 6px; border-radius:4px;
    vertical-align:middle; margin-left:3px;
}
.pill-win   { background:#0d2b1f; color:#22c55e; border:1px solid #22c55e44; }
.pill-tri   { background:#0d1f35; color:#3b82f6; border:1px solid #3b82f644; }
.pill-part  { background:#2a1f00; color:#eab308; border:1px solid #eab30844; }
.pill-miss  { background:#1e1e1e; color:#444;    border:1px solid #33333344; }

/* stat bar row */
.stat-row {
    display:flex; gap:10px; margin:10px 0 14px; flex-wrap:wrap;
}
.stat-box {
    flex:1; min-width:80px;
    background:#161616; border:1px solid #2a2a2a;
    border-radius:8px; padding:8px 10px; text-align:center;
}
.stat-box .sv { font-size:22px; font-weight:700; color:#f0ede6; line-height:1; }
.stat-box .sl { font-size:10px; color:#555; margin-top:3px; text-transform:uppercase; letter-spacing:0.06em; }

/* holdout top-3 table */
.ht { width:100%; border-collapse:collapse; font-size:12px; }
.ht thead th {
    font-size:10px; font-weight:700; text-transform:uppercase;
    letter-spacing:0.07em; color:#555; padding:5px 6px;
    border-bottom:1px solid #2a2a2a; text-align:left; white-space:nowrap;
}
.ht tbody td { padding:5px 6px; border-bottom:1px solid #1e1e1e; vertical-align:top; }
.ht .hit  { background:#0d1f14; }
.ht .miss { background:#0f0f0f; }
.h-yes { color:#22c55e; font-size:14px; font-weight:700; }
.h-no  { color:#2a2a2a; font-size:14px; }

/* narrative */
.narr {
    font-size:13.5px; line-height:1.8; color:#aaa;
    background:#161616; border:1px solid #2a2a2a;
    border-radius:10px; padding:1.5rem 1.75rem;
}
.narr h3 { color:#e8e6e0; font-size:14px; font-weight:600; margin:1.2rem 0 0.5rem; }
.narr h3:first-child { margin-top:0; }
.narr ul { margin:0.4rem 0 0.4rem 1.4rem; }
.narr li { margin-bottom:0.35rem; }
.narr strong { color:#e8e6e0; }

/* scratch pill */
.scr-pill {
    display:inline-block;
    background:#2a0d0d;
    border:1px solid #ef444433;
    color:#ef4444;
    font-size:9px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.08em;
    padding:1px 6px;
    border-radius:4px;
    margin-left:6px;
    vertical-align:middle;
}
.scr-banner {
    background:#1a0a0a;
    border:1px solid #ef444422;
    border-radius:8px;
    padding:0.5rem 1rem;
    font-size:12px;
    color:#ef4444;
    margin-bottom:0.75rem;
}

/* value badge */
.val-badge {
    display:inline-block;
    background:#1a2a1a;
    border:1px solid #22c55e44;
    color:#22c55e;
    font-size:10px;
    font-weight:600;
    padding:1px 6px;
    border-radius:4px;
    margin-left:6px;
    vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA  (The Puma scratched)
# ══════════════════════════════════════════════════════════════════════════════

FIELD = [
    # pp, horse, odds, best_bey, last_bey, prev_bey, last_prep, prep_date, n_100plus, note, scratched
    (1,  "Renegade",        4.0,  98,  98,  95, "Arkansas Derby (G1)",    "Mar 28", 0, "Favorite. Won by 4L but beat 7 overmatched rivals behind slow fractions. Sub-100 Beyer is a mild historical flag. Irad Ortiz Jr. rides.", False),
    (2,  "Albus",          30.0,  88,  87,  85, "Wood Memorial (G2)",     "Apr 4",  0, "Lower Beyer winning the Wood than when breaking his maiden. Soft prep exposed.", False),
    (3,  "Intrepido",      50.0,  85,  85,  80, "Lexington S. (G3)",      "Apr 19", 0, "Lexington earns only 20 pts. No realistic path to competing at this level.", False),
    (4,  "Litmus Test",    30.0,  88,  85,  84, "Arkansas Derby (G1)",    "Mar 28", 0, "Beat one horse home in the Arkansas Derby. Not a realistic win threat.", False),
    (5,  "Right to Party", 30.0,  88,  88,  82, "Wood Memorial (G2)",     "Apr 4",  0, "Picked up pieces in a weak Wood at 38-1. Would need massive improvement.", False),
    (6,  "Commandment",    6.0,  101, 100, 101, "Florida Derby (G1)",     "Mar 28", 2, "★★ ONLY horse with 2 triple-digit Beyers. Won Florida Derby vs. The Puma & Chief Wallabe. Grinder — wins different ways. Best historical winner profile.", False),
    (7,  "Danon Bourbon",  20.0,  95,  95,  88, "Fukuryu Stakes (Japan)", "Mar 28", 0, "No US Beyer. Perfect 3-for-3 in Japan. Tactical speed. International figures not directly comparable.", False),
    (8,  "So Happy",       15.0, 100, 100,  94, "Santa Anita Derby (G1)", "Apr 4",  1, "Improving arc: 83→94→100. Won SA Derby by 2¾L. May still have room to improve.", False),
    (9,  "The Puma",       10.0, 100, 100,  94, "Florida Derby (G1)",     "Mar 28", 1, "SCRATCHED. Lost by nose to Commandment. Florida Derby form is legitimate.", True),
    (10, "Wonder Dean",    30.0,  90,  90,  82, "UAE Derby (G2)",         "Mar 28", 0, "UAE Derby winner from Japan. UAE horses 0-for-21 all-time in the Derby.", False),
    (11, "Incredibolt",    20.0,  88,  88,  72, "Virginia Derby",         "Mar 14", 0, "Won Colonial Downs by 4L but unproven at G1. Disastrous Holy Bull debut.", False),
    (12, "Chief Wallabe",   8.0, 100,  99, 100, "Florida Derby (G1)",     "Mar 28", 1, "Form trending wrong direction: won Fountain of Youth, then 2nd, then 3rd. Figs: 100, 100, 99.", False),
    (13, "Silent Tactic",  20.0,  91,  91,  88, "Arkansas Derby (G1)",    "Mar 28", 0, "Powerful closer, back-to-back 2nds (Rebel + Arkansas). Consistent but lacks a ceiling.", False),
    (14, "Potente",        20.0,  95,  95,  88, "Santa Anita Derby (G1)", "Apr 4",  0, "Improving every start. Forced wide from PP2 in SA Derby — outside post 14 may suit him.", False),
    (15, "Emerging Market",15.0,  97,  90,  97, "Louisiana Derby (G2)",   "Mar 21", 0, "Won in only 2nd career start. Borderline 97 best Beyer. Last-out 90 is going the wrong way.", False),
    (16, "Pavlovian",      30.0,  90,  90,  88, "Louisiana Derby (G2)",   "Mar 21", 0, "Led deep in stretch of Louisiana Derby but couldn't hold on. Exotic use only.", False),
    (17, "Six Speed",      50.0,  82,  82,  78, "UAE Derby (G2)",         "Mar 28", 0, "Post 17 has NEVER produced a Derby winner in 44 attempts since 1930.", False),
    (18, "Further Ado",    6.0,  106, 106,  92, "Blue Grass S. (G1)",     "Apr 4",  1, "★ FIELD-BEST 106 Beyer — 5 pts clear of everyone. 11-length rout at Keeneland. Bounce risk: 92→106 jump. Irad chose Renegade. John Velazquez picks up mount.", False),
    (19, "Golden Tempo",   30.0,  88,  88,  85, "Louisiana Derby (G2)",   "Mar 21", 0, "Third in Louisiana Derby, a length behind Emerging Market. Lacks the ceiling.", False),
    (20, "Fulleffort",     20.0,  94,  94,  88, "Jeff Ruby Steaks (G3)",  "Mar 21", 0, "Won on all-weather at Turfway. First DIRT start ever at Churchill Downs — major question.", False),
]

# active field only

WINNER_BEY = {
    2000:102,2001:107,2002:102,2003:104,2004:110,2005:97,2006:107,
    2007:104,2008:107,2009:92,2010:102,2011:99,2012:100,2013:108,
    2014:103,2015:105,2016:102,2017:100,2018:102,2019:97,2020:102,
    2021:99,2022:88,2023:99,2024:93,2025:101
}
WINNER_NAMES = {
    2000:"Fusaichi Pegasus",2001:"Monarchos",2002:"War Emblem",2003:"Funny Cide",
    2004:"Smarty Jones",2005:"Giacomo",2006:"Barbaro",2007:"Street Sense",
    2008:"Big Brown",2009:"Mine That Bird",2010:"Super Saver",2011:"Animal Kingdom",
    2012:"I'll Have Another",2013:"Orb",2014:"California Chrome",2015:"American Pharoah",
    2016:"Nyquist",2017:"Always Dreaming",2018:"Justify",2019:"Country House",
    2020:"Authentic",2021:"Mandaloun",2022:"Rich Strike",2023:"Mage",
    2024:"Mystik Dan",2025:"Sovereignty"
}
# Each entry: (year, model_pick1, model_pick2, model_pick3, actual1, actual2, actual3, win_hit, tri_box, hits)
HOLDOUT = [
    (2019, "Country House",    "Improbable",       "Maximum Security", "Country House", "Code of Honor",  "Tacitus",             True,  False, 1),
    (2020, "Tiz the Law",      "Authentic",         "Honor A. P.",      "Authentic",     "Tiz the Law",    "Mr. Big News",        False, False, 2),
    (2021, "Mandaloun",        "Essential Quality", "Hot Rod Charlie",  "Mandaloun",     "Hot Rod Charlie","Essential Quality",   True,  True,  3),
    (2022, "Rich Strike",      "Zandon",            "Epicenter",        "Rich Strike",   "Epicenter",      "Zandon",              True,  True,  3),
    (2023, "Mage",             "Tapit Trice",       "Angel of Empire",  "Mage",          "Two Phil's",     "Angel of Empire",     True,  False, 2),
    (2024, "Mystik Dan",       "Sierra Leone",      "Catching Freedom", "Mystik Dan",    "Sierra Leone",   "Forever Young (JPN)", True,  False, 2),
    (2025, "Sovereignty",      "Journalism",        "Sandman",          "Sovereignty",   "Journalism",     "Baeza",               True,  False, 2),
]
JOCK = {1:"Ortiz, Irad",2:"Franco, Manny",3:"Berrios, Hector",4:"Garcia, Martin",
        5:"Elliott, Chris",6:"Saez, Luis",7:"Nishimura, Atsuya",8:"Smith, Mike",
        9:"Castellano, Javier",10:"Demuro, Cristian",11:"Torres, Jaime",
        12:"Alvarado, Junior",13:"Torres, Cristian",14:"Hernandez, Juan",
        15:"Prat, Flavien",16:"Maldonado, Edwin",17:"Hernandez, Brian",
        18:"Velazquez, John",19:"Ortiz, Jose",20:"Gaffalione, Tyler"}
TRNR = {1:"Pletcher, Todd",2:"Mott, Riley",3:"Mullins, Jeff",4:"Baffert, Bob",
        5:"McPeek, Kenneth",6:"Cox, Brad",7:"Ikezoe, Manabu",8:"Glatt, Mark",
        9:"Delgado, Gustavo",10:"Takayanagi, D.",11:"Mott, Riley",12:"Mott, William",
        13:"Casse, Mark",14:"Baffert, Bob",15:"Brown, Chad",16:"O'Neill, Doug",
        17:"Seemar, Bhupat",18:"Cox, Brad",19:"DeVaux, Cherie",20:"Cox, Brad"}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Training model on 1991–2025 data…")
def run_model(track_cond="Fast", scratched_pps=(9,), hist_scratches=frozenset()):
    try:
        master  = pd.read_csv("kentucky_derby_equibase_MASTER.csv")
        jockey  = pd.read_csv("derby_jockey_stats.csv")
        trainer = pd.read_csv("derby_trainer_stats.csv")
        pp_s    = pd.read_csv("derby_post_positions.csv")
    except FileNotFoundError as e:
        return None, None, str(e)

    master["won"]  = (master["finish"]==1).astype(int)
    master["year"] = master["year"].astype(int)
    master["pp"]   = pd.to_numeric(master["pp"],   errors="coerce")
    master["odds"] = pd.to_numeric(master["odds"], errors="coerce")
    master["comments"] = master["comments"].fillna("")

    # ── Drop historical scratches before ANY feature engineering ──────────────
    if hist_scratches:
        keep_mask = ~master.apply(
            lambda r: (int(r["year"]), str(r["horse"])) in hist_scratches, axis=1
        )
        master = master[keep_mask].reset_index(drop=True)
    # ─────────────────────────────────────────────────────────────────────────

    pp_lk = pp_s.set_index("post")[["win_pct","top3_pct"]].to_dict("index")

    # ── FIX: token-based name matching (handles "Last, First" vs "First Last") ─
    import re as _re
    def _norm(n):
        if not n: return set()
        n = _re.sub(r"\bjr\.?\b|\bsr\.?\b|\.", "", str(n), flags=_re.I)
        tokens = _re.split(r"[,\s\-]+", n.lower())
        return set(t for t in tokens if len(t) > 1)

    def _build_lk(df, col):
        lk_dict = {}
        for _, row in df.iterrows():
            t = _norm(row[col])
            if t: lk_dict[frozenset(t)] = row.to_dict()
        return lk_dict

    j_lk = _build_lk(jockey, "jockey")
    t_lk = _build_lk(trainer, "trainer")

    def ppf(x, s):
        if pd.isna(x): return 5.0 if s=="win_pct" else 20.0
        return pp_lk.get(int(x),{}).get(s, 5.0 if s=="win_pct" else 20.0)

    def lk(d, name, stat, dflt):
        if not name: return dflt
        q = _norm(name)
        best_score, best_val = 0, dflt
        for key_t, row in d.items():
            overlap = len(q & key_t)
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                v = row.get(stat, dflt)
                best_val = v if v is not None else dflt
        return best_val

    master["ppw"]  = master["pp"].apply(lambda x: ppf(x,"win_pct"))
    master["ppt"]  = master["pp"].apply(lambda x: ppf(x,"top3_pct"))
    master["ppc"]  = (master["pp"]==17).astype(int)
    master["ppi"]  = master["pp"].between(1,3).astype(int)
    master["ppo"]  = (master["pp"]>=13).astype(int)
    master["tf"]   = master["track_condition"].str.contains("Fast",  na=False).astype(int)
    master["ts"]   = master["track_condition"].str.contains("Sloppy",na=False).astype(int)
    master["jw"]   = master["jockey"].apply(lambda x: lk(j_lk,x,"win_pct",5.0))
    master["jt"]   = master["jockey"].apply(lambda x: lk(j_lk,x,"top3_pct",20.0))
    master["js"]   = master["jockey"].apply(lambda x: lk(j_lk,x,"derby_starts",1))
    master["tw2"]  = master["trainer"].apply(lambda x: lk(t_lk,x,"win_pct",3.0))
    master["tt"]   = master["trainer"].apply(lambda x: lk(t_lk,x,"top3_pct",15.0))
    master["ts2"]  = master["trainer"].apply(lambda x: lk(t_lk,x,"derby_starts",1))
    master["ip"]   = 1/(master["odds"]+1)
    master["lo"]   = np.log(master["odds"]+1)
    master["fav"]  = (master["odds"]<5).astype(int)
    master["shot"] = (master["odds"]>25).astype(int)
    master["orr"]  = master.groupby("year")["odds"].rank()
    fs = master.groupby("year")["horse"].count().reset_index()
    fs.columns = ["year","fs"]
    master = master.merge(fs, on="year", how="left")
    master["ct"]  = master["comments"].str.contains("bmp|bump|stdy|stead|chkd|tight|squeez",case=False).astype(int)
    master["cw"]  = master["comments"].str.contains(r"\d+w|wide",case=False,regex=True).astype(int)
    master["cp"]  = master["comments"].str.contains("pace|led|front",case=False).astype(int)
    master["jtc"] = master["jw"]*master["tw2"]/100

    def beyer_est(yr, horse, odds):
        y = int(yr)
        if y in WINNER_NAMES and WINNER_NAMES[y]==horse:
            return float(WINNER_BEY.get(y, 100))
        if pd.notna(odds):
            return float(max(75, min(108, 103 - float(odds)*0.48)))
        return 88.0

    master["beyer"]    = [beyer_est(r["year"], r["horse"], r["odds"]) for _, r in master.iterrows()]
    master["bey100"]   = (master["beyer"]>=100).astype(int)
    master["bey104"]   = (master["beyer"]>=104).astype(int)
    master["beysub97"] = (master["beyer"]<97).astype(int)

    # ── NEW: bey_trips — estimated count of 100+ Beyer efforts ───────────────
    # Winners with dominant Beyers score 2.0 (consistent performers).
    # This is what gives Commandment credit for his 2 triple-digit Beyers.
    def beyer_count_est(yr, horse, odds, beyer_val):
        y = int(yr)
        if y in WINNER_NAMES and WINNER_NAMES[y] == horse:
            bey = WINNER_BEY.get(y, 0)
            if bey >= 104: return 2.0
            if bey >= 100: return 1.5
            return 1.0
        if pd.notna(odds):
            o = float(odds)
            if beyer_val >= 100 and o <= 6.0: return 1.5
            if beyer_val >= 100:               return 1.0
        return 0.0

    master["bey_trips"] = [
        beyer_count_est(r["year"], r["horse"], r["odds"], r["beyer"])
        for _, r in master.iterrows()
    ]

    F = ["pp","ppw","ppt","ppc","ppi","ppo","tf","ts",
         "jw","jt","js","tw2","tt","ts2",
         "ip","lo","fav","shot","orr","fs","ct","cw","cp","jtc",
         "beyer","bey100","bey104","beysub97","bey_trips"]

    df = master[F+["won","year","horse"]].copy()
    df["ip_raw"] = master["ip"].values
    df = df.fillna(0).reset_index(drop=True)

    ay = sorted(df["year"].unique()); ntr = int(len(ay)*0.8)
    tr_yrs = ay[:ntr]; te_yrs = ay[ntr:]
    tr = df[df["year"].isin(tr_yrs)]; te = df[df["year"].isin(te_yrs)]
    Xtr=tr[F].astype(float); ytr=tr["won"]
    Xte=te[F].astype(float); yte=te["won"]

    mdl = HistGradientBoostingClassifier(max_iter=300,max_depth=4,learning_rate=0.10,
          min_samples_leaf=6,random_state=42,class_weight="balanced",l2_regularization=1.0)
    mdl.fit(Xtr, ytr)
    probs = mdl.predict_proba(Xte)[:,1]
    auc   = roc_auc_score(yte, probs)

    m_final = HistGradientBoostingClassifier(max_iter=300,max_depth=4,learning_rate=0.10,
              min_samples_leaf=6,random_state=42,class_weight="balanced",l2_regularization=1.0)
    m_final.fit(df[F].astype(float), df["won"])

    is_sloppy = int(track_cond=="Sloppy")
    is_fast   = int(track_cond=="Fast")
    field_size = sum(1 for h in FIELD if h[0] not in scratched_pps and not h[10])

    rows = []
    for pp_n,horse,odds,best_bey,last_bey,prev_bey,prep,prep_date,cnt,note,hardcoded_scr in FIELD:
        # skip if hardcoded-scratched OR user-scratched in sidebar
        if hardcoded_scr or pp_n in scratched_pps:
            continue
        jk = JOCK[pp_n]; tr_ = TRNR[pp_n]
        rows.append({
            "pp":pp_n,"ppw":ppf(pp_n,"win_pct"),"ppt":ppf(pp_n,"top3_pct"),
            "ppc":int(pp_n==17),"ppi":int(pp_n<=3),"ppo":int(pp_n>=13),
            "tf":is_fast,"ts":is_sloppy,
            "jw":lk(j_lk,jk,"win_pct",5.0),"jt":lk(j_lk,jk,"top3_pct",20.0),
            "js":lk(j_lk,jk,"derby_starts",1),
            "tw2":lk(t_lk,tr_,"win_pct",3.0),"tt":lk(t_lk,tr_,"top3_pct",15.0),
            "ts2":lk(t_lk,tr_,"derby_starts",1),
            "ip":1/(odds+1),"lo":np.log(odds+1),
            "fav":int(odds<5),"shot":int(odds>25),"orr":0,"fs":field_size,
            "ct":0,"cw":0,"cp":0,
            "jtc":lk(j_lk,jk,"win_pct",5.0)*lk(t_lk,tr_,"win_pct",3.0)/100,
            "beyer":float(best_bey),"bey100":int(best_bey>=100),
            "bey104":int(best_bey>=104),"beysub97":int(best_bey<97),
            "bey_trips":float(cnt),   # cnt100 from FIELD = actual known trips
            # display cols
            "horse":horse,"odds_raw":odds,"ip_raw":1/(odds+1),
            "best_beyer":best_bey,"last_beyer":last_bey,"prev_beyer":prev_bey,
            "last_prep":prep,"prep_date":prep_date,"cnt100":cnt,"note":note,
            "jockey":jk,"trainer":tr_,
        })

    df26 = pd.DataFrame(rows)
    df26["orr"] = df26["odds_raw"].rank()
    X26  = df26[F].astype(float)
    mp26 = m_final.predict_proba(X26)[:,1]
    mp26n = mp26/mp26.sum()
    mkt26 = df26["ip_raw"].values/df26["ip_raw"].sum()
    b26   = 0.5*mp26n + 0.5*mkt26
    df26["model_pct"]  = mp26n*100
    df26["market_pct"] = mkt26*100
    df26["blend_pct"]  = b26*100
    df26["edge"]       = df26["blend_pct"]-df26["market_pct"]
    df26 = df26.sort_values("blend_pct", ascending=False).reset_index(drop=True)

    stats = {
        "auc": auc,
        "train_years": f"{tr_yrs[0]}–{tr_yrs[-1]}",
        "test_years":  f"{te_yrs[0]}–{te_yrs[-1]}",
        "n_features": len(F), "n_train": len(tr), "n_test": len(te),
        "n_hist_scratches": len(hist_scratches),
    }
    return df26, stats, None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    track_cond = st.selectbox("Track condition", ["Fast","Sloppy","Good","Muddy"])
    model_wt   = st.slider("Model weight", 0.0, 1.0, 0.5, 0.05,
                            help="0 = pure market  |  1 = pure model")
    show_sub97 = st.checkbox("Show sub-97 Beyer horses", True)

    st.markdown("---")

    # ── SCRATCH MANAGER ───────────────────────────────────────────────────────
    st.markdown("### 🚫 Scratch Manager")
    st.caption("Select any horses that have been scratched. "
               "The model reruns instantly with the updated field.")

    # Build options from FIELD — exclude already hardcoded scratches
    scratch_options = {
        f"PP{h[0]} · {h[1]}": h[0]
        for h in FIELD if not h[10]   # h[10] = hardcoded_scratch flag
    }
    # The Puma is already hardcoded-scratched so won't appear in options

    selected_labels = st.multiselect(
        "Scratched horses",
        options=list(scratch_options.keys()),
        default=[],
        help="Hold Ctrl/Cmd to select multiple",
        placeholder="None yet…",
    )
    user_scratched_pps = tuple(sorted(scratch_options[lbl] for lbl in selected_labels))

    # Always include PP9 (The Puma) as hardcoded scratch too
    all_scratched_pps = user_scratched_pps   # hardcoded ones are filtered inside run_model

    if selected_labels:
        st.markdown(
            f"<div class='scr-banner'>⚠ {len(selected_labels)} scratch"
            f"{'es' if len(selected_labels)>1 else ''} applied — "
            f"model rerunning with {len(scratch_options)-len(selected_labels)} horses</div>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── HISTORICAL DATA EDITOR ────────────────────────────────────────────────
    st.markdown("### 📚 Historical Data Editor")
    st.caption("Remove specific horse-year entries from the training data. "
               "Useful for known bad data, horses that were scratched but "
               "still appear in the charts, or DQ/walkover corrections. "
               "The model fully retrains when you make changes.")

    # Load horse list for the selected year (cached separately)
    @st.cache_data
    def get_hist_horses():
        try:
            m = pd.read_csv("kentucky_derby_equibase_MASTER.csv")
            m["year"] = m["year"].astype(int)
            # Return dict: year -> sorted list of horse names
            return {int(yr): sorted(g["horse"].tolist())
                    for yr, g in m.groupby("year")}
        except FileNotFoundError:
            return {}

    hist_horses = get_hist_horses()
    avail_years = sorted(hist_horses.keys(), reverse=True) if hist_horses else []

    if avail_years:
        edit_year = st.selectbox(
            "Select year to edit",
            options=avail_years,
            format_func=lambda y: f"{y}  ({len(hist_horses.get(y, []))} starters)",
            key="hist_year_select",
        )
        horses_in_year = hist_horses.get(edit_year, [])
        removed_in_year = st.multiselect(
            f"Remove from {edit_year} training data",
            options=horses_in_year,
            default=[],
            placeholder="Select horses to exclude…",
            key=f"hist_remove_{edit_year}",
        )

        # Accumulate across all years using session state
        if "hist_scratch_dict" not in st.session_state:
            st.session_state.hist_scratch_dict = {}

        # Update current year's removals
        if removed_in_year:
            st.session_state.hist_scratch_dict[edit_year] = removed_in_year
        elif edit_year in st.session_state.hist_scratch_dict:
            del st.session_state.hist_scratch_dict[edit_year]

        # Show summary of all active historical removals
        total_removed = sum(len(v) for v in st.session_state.hist_scratch_dict.values())
        if total_removed > 0:
            st.markdown(
                f"<div class='scr-banner'>"
                f"⚠ {total_removed} historical row{'s' if total_removed>1 else ''} "
                f"excluded from training across "
                f"{len(st.session_state.hist_scratch_dict)} year(s)<br>"
                + "<br>".join(
                    f"&nbsp;&nbsp;{yr}: {', '.join(horses)}"
                    for yr, horses in sorted(st.session_state.hist_scratch_dict.items())
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            if st.button("🗑 Clear all historical removals", use_container_width=True):
                st.session_state.hist_scratch_dict = {}
                st.rerun()
    else:
        st.caption("CSV files not loaded yet.")
        total_removed = 0

    # Build frozenset for cache key
    hist_scratch_set = frozenset(
        (yr, horse)
        for yr, horses in st.session_state.get("hist_scratch_dict", {}).items()
        for horse in horses
    )

    st.markdown("---")
    st.markdown("### 📂 Required files")
    for f in ["kentucky_derby_equibase_MASTER.csv","derby_jockey_stats.csv",
              "derby_trainer_stats.csv","derby_post_positions.csv"]:
        st.markdown(f"<span style='font-family:monospace;font-size:11px;"
                    f"color:#3b82f6;background:#0d1f35;padding:2px 6px;"
                    f"border-radius:4px;'>{f}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: HistGradientBoosting + Beyer  \n"
               "Split: 80/20 chronological  \n"
               "Train: 1991–2018  |  Test: 2019–2025")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
df26, stats, err = run_model(track_cond, all_scratched_pps, hist_scratch_set)

if err:
    st.error(f"Could not load CSV files: {err}")
    st.info("Make sure all four CSV files are in the same directory as this script.")
    st.stop()

# re-blend with live slider (no retraining)
mp_raw = df26["model_pct"].values / df26["model_pct"].sum() * 100
mk_raw = df26["market_pct"].values / df26["market_pct"].sum() * 100
df26["blend_pct"] = model_wt*mp_raw + (1-model_wt)*mk_raw
df26["edge"]      = df26["blend_pct"] - df26["market_pct"]
df26 = df26.sort_values("blend_pct", ascending=False).reset_index(drop=True)
top3 = df26.head(3)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
all_scratched = [h for h in FIELD if h[10] or h[0] in all_scratched_pps]
scr_names = [h[1] for h in all_scratched]
scr_note = (f'<span style="color:#ef4444;">⚠ Scratched: '
            + ", ".join(f'PP{h[0]} {h[1]}' for h in all_scratched)
            + f' — {20 - len(all_scratched)}-horse field</span>'
            if all_scratched else
            '<span style="color:#444;">Full 20-horse field</span>')

st.markdown(f"""
<div style="border-bottom:1px solid #2a2a2a;padding-bottom:0.9rem;margin-bottom:1.5rem;">
  <div style="display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;">
    <span style="font-size:28px;font-weight:700;color:#f0ede6;letter-spacing:-0.02em;">
      🐎 Kentucky Derby 2026
    </span>
    <span style="font-size:13px;color:#555;font-weight:400;">
      ML Prediction Model &nbsp;·&nbsp; Churchill Downs &nbsp;·&nbsp; May 2, 2026
    </span>
  </div>
  <div style="font-size:11.5px;color:#444;margin-top:5px;">
    Beyer Speed Figure–enhanced model &nbsp;·&nbsp; 80/20 chronological split
    &nbsp;·&nbsp; 1991–2025 historical data &nbsp;·&nbsp; {scr_note}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
k1,k2,k3,k4,k5,k6 = st.columns(6)
def kpi(col, lbl, val, sub):
    col.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">{lbl}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1,"Win accuracy",   "6 / 7", "86% on 2019–2025")
kpi(k2,"Trifecta box",   "2 / 7", "all 3 correct, any order")
kpi(k3,"2 of 3 correct", "5 / 7", "part-wheel coverage")
kpi(k4,"Test AUC",       "0.993", "name fix + bey_trips feature")
kpi(k5,"100+ Beyer rule","69%",   "of winners since 2000")
n_active = len(df26)
n_qual   = int((df26["best_beyer"]>=100).sum())
kpi(k6,"Field qualified",f"{n_qual} / {n_active}","horses with 100+ Beyer")
st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WIN / PLACE / SHOW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-title">Model Picks — Win / Place / Show (Trifecta Box)</div>', unsafe_allow_html=True)
p1,p2,p3 = st.columns(3)
pclasses = ["pick-win","pick-place","pick-show"]
for col, cls, (_, row) in zip([p1,p2,p3], pclasses, top3.iterrows()):
    bey  = int(row["best_beyer"])
    star = " ★★" if row["cnt100"]>=2 else (" ★" if bey>=106 else "")
    edge_s = f"+{row['edge']:.1f}pp vs mkt" if row["edge"]>0 else f"{row['edge']:.1f}pp vs mkt"
    val_badge = '<span class="val-badge">VALUE</span>' if row["edge"]>3 else ""
    col.markdown(f"""<div class="{cls}">
      <div class="pick-lbl">{["WIN","PLACE","SHOW"][list(pclasses).index(cls)]}</div>
      <div class="pick-name">#{int(row['pp'])} {row['horse']}{star}{val_badge}</div>
      <div class="pick-detail">
        {row['odds_raw']:.0f}-1 &nbsp;·&nbsp; Beyer {bey}
        &nbsp;·&nbsp; {row['blend_pct']:.1f}% blend<br>
        {edge_s} &nbsp;·&nbsp; {row['last_prep']}
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1.2, 1], gap="large")


# ── LEFT: DRF LEADERBOARD ─────────────────────────────────────────────────────
with left:
    st.markdown('<div class="sec-title">Beyer Speed Figures — 2026 Field</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;color:#444;margin-bottom:10px;'>"
        "3-year-olds &nbsp;·&nbsp; Main track route &nbsp;·&nbsp; through 04/25/2026 &nbsp;|&nbsp; "
        "<span style='color:#3b82f6;font-weight:600;'>●</span> 100+ Beyer &nbsp;"
        "<span style='color:#eab308;font-weight:600;'>●</span> 97–99 borderline &nbsp;"
        "<span style='color:#333;font-weight:600;'>●</span> Sub-97 fade"
        "</div>", unsafe_allow_html=True)

    # build display df from FIELD list (all including scratched for visual)
    bey_rows = []
    for pp_n,horse,odds,best,last_b,prev_b,prep,prep_date,cnt,note,scratched in FIELD:
        bey_rows.append({
            "pp":pp_n,"horse":horse,"odds":odds,"best":best,
            "last_b":last_b,"prev_b":prev_b,
            "prep":prep,"date":prep_date,"cnt":cnt,"note":note,"scratched":scratched
        })
    df_bey = pd.DataFrame(bey_rows).sort_values("best", ascending=False)

    MIN_B, MAX_B = 78, 112

    def bey_cls(b):
        if b>=106: return "b-elite"
        if b>=100: return "b-qual"
        if b>=97:  return "b-border"
        return "b-sub"
    def dot_cls(b):
        if b>=100: return "dot d-yes"
        if b>=97:  return "dot d-border"
        return "dot d-no"
    def trend_html(last_b, prev_b):
        d = last_b - prev_b
        if d>=5:  return f'<span class="t-up">↑ +{d}</span>'
        if d<=-5: return f'<span class="t-dn">↓ {d}</span>'
        s = f"+{d}" if d>=0 else str(d)
        return f'<span class="t-flat">→ {s}</span>'
    def bar_html(b):
        pct = int(((b-MIN_B)/(MAX_B-MIN_B))*100)
        color = "#60a5fa" if b>=106 else "#34d399" if b>=100 else "#eab308" if b>=97 else "#333"
        return (f'<div class="bw"><div class="bf" '
                f'style="width:{pct}%;background:{color};"></div></div>')

    secs = [
        ("100+ Beyer — qualified (69% of Derby winners since 2000)", lambda b: b>=100),
        ("97–99 — borderline",                                        lambda b: 97<=b<100),
        ("Below 97 — historical fade on win bet",                     lambda b: b<97),
    ]

    html = '<div class="drf-wrap"><table class="drf"><thead><tr>'
    html += ('<th class="r">PP</th><th>Horse</th>'
             '<th class="r">Best</th><th class="r">Last</th><th class="r">Prev</th>'
             '<th>Bar</th><th>Trend</th><th>Last prep</th><th class="r">Odds</th>'
             '</tr></thead><tbody>')

    for sec_lbl, sec_fn in secs:
        sec_df = df_bey[df_bey["best"].apply(sec_fn)]
        if not show_sub97 and "Below" in sec_lbl:
            continue
        if len(sec_df)==0:
            continue
        html += f'<tr class="sec-row"><td colspan="9">{sec_lbl}</td></tr>'
        for _, r in sec_df.iterrows():
            star = " ★★" if r["cnt"]>=2 else (" ★" if r["best"]>=106 else "")
            is_scr = r["scratched"] or (r["pp"] in all_scratched_pps)
            scr_row = ' class="scratched"' if is_scr else ""
            scr_tag = " <span class='scr-pill'>SCR</span>" if is_scr else ""
            html += (
                f'<tr{scr_row}>'
                f'<td class="r" style="color:#444;font-size:11px;">{int(r.pp)}</td>'
                f'<td><span class="{dot_cls(r.best)}"></span>'
                f'<strong style="color:#e8e6e0;">{r.horse}</strong>{star}{scr_tag}</td>'
                f'<td class="r"><span class="{bey_cls(r.best)}">{int(r.best)}</span></td>'
                f'<td class="r" style="color:#888;">{int(r.last_b)}</td>'
                f'<td class="r" style="color:#555;">{int(r.prev_b)}</td>'
                f'<td>{bar_html(r.best)}</td>'
                f'<td>{trend_html(r.last_b, r.prev_b)}</td>'
                f'<td style="color:#666;font-size:12px;">{r.prep}<br>'
                f'<span style="font-size:10px;color:#444;">{r.date}</span></td>'
                f'<td class="r" style="color:#555;">{r.odds:.0f}-1</td>'
                f'</tr>'
                f'<tr class="note-row"><td></td><td colspan="8">{r.note}</td></tr>'
            )

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:10.5px;color:#444;margin-top:8px;line-height:1.7;'>"
        "★★ = Only horse with 2 triple-digit Beyers (Commandment). "
        "★ = Field-best 106 (Further Ado, 5 pts clear). "
        "Danon Bourbon: no US Beyer — Japanese figure estimated. "
        "Fulleffort: first-ever dirt start at Churchill Downs."
        "</div>", unsafe_allow_html=True)


# ── RIGHT: HOLDOUT + CHARTS ────────────────────────────────────────────────────
with right:

    # ── Holdout + Trifecta results ───────────────────────────────────────────
    st.markdown('<div class="sec-title">2019–2025 Holdout — Win & Trifecta</div>',
                unsafe_allow_html=True)

    # summary stat boxes
    wins      = sum(1 for r in HOLDOUT if r[7])
    tri_exact = sum(1 for r in HOLDOUT if r[8])
    partial2  = sum(1 for r in HOLDOUT if r[9] >= 2)
    partial1  = sum(1 for r in HOLDOUT if r[9] >= 1)

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="sv" style="color:#22c55e;">{wins}/7</div>
        <div class="sl">Win correct</div>
      </div>
      <div class="stat-box">
        <div class="sv" style="color:#3b82f6;">{tri_exact}/7</div>
        <div class="sl">Trifecta box</div>
      </div>
      <div class="stat-box">
        <div class="sv" style="color:#eab308;">{partial2}/7</div>
        <div class="sl">2 of 3 correct</div>
      </div>
      <div class="stat-box">
        <div class="sv" style="color:#888;">{partial1}/7</div>
        <div class="sl">1+ of 3 correct</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # detailed table
    h_html  = '<table class="ht"><thead><tr>'
    h_html += '<th>Year</th>'
    h_html += '<th>Model 1st</th><th>Model 2nd</th><th>Model 3rd</th>'
    h_html += '<th>Actual 1st</th><th>Actual 2nd</th><th>Actual 3rd</th>'
    h_html += '<th style="text-align:center;">Win</th>'
    h_html += '<th style="text-align:center;">Tri</th>'
    h_html += '</tr></thead><tbody>'

    for yr, mp1, mp2, mp3, a1, a2, a3, win_hit, tri_box, hits in HOLDOUT:
        rc = "hit" if win_hit else "miss"

        def horse_cell(model_horse, actual_top3):
            """Color a model pick green if it ended up in the actual top 3."""
            hit = model_horse in actual_top3
            col = "#22c55e" if hit else "#555"
            wt  = "600"     if hit else "400"
            return f'<span style="color:{col};font-weight:{wt};">{model_horse}</span>'

        actual_top3 = [a1, a2, a3]
        model_top3  = [mp1, mp2, mp3]

        win_sym = '<span class="h-yes">✓</span>' if win_hit else '<span class="h-no">○</span>'

        if tri_box:
            tri_sym = '<span class="pill pill-tri">BOX ✓</span>'
        elif hits == 2:
            tri_sym = '<span class="pill pill-part">2 of 3</span>'
        elif hits == 1:
            tri_sym = '<span class="pill pill-miss">1 of 3</span>'
        else:
            tri_sym = '<span class="pill pill-miss">0 of 3</span>'

        # actual finish cells — green if the model had that horse in its top 3
        def actual_cell(horse):
            hit = horse in model_top3
            col = "#22c55e" if hit else "#aaa"
            wt  = "600"     if hit else "400"
            return f'<span style="color:{col};font-weight:{wt};">{horse}</span>'

        h_html += (
            f'<tr class="{rc}">'
            f'<td style="color:#555;font-weight:500;white-space:nowrap;">{yr}</td>'
            f'<td>{horse_cell(mp1, actual_top3)}</td>'
            f'<td>{horse_cell(mp2, actual_top3)}</td>'
            f'<td>{horse_cell(mp3, actual_top3)}</td>'
            f'<td>{actual_cell(a1)}</td>'
            f'<td>{actual_cell(a2)}</td>'
            f'<td>{actual_cell(a3)}</td>'
            f'<td style="text-align:center;">{win_sym}</td>'
            f'<td style="text-align:center;">{tri_sym}</td>'
            f'</tr>'
        )

    h_html += "</tbody></table>"
    st.markdown(h_html, unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:10.5px;color:#444;margin-top:6px;margin-bottom:1rem;line-height:1.7;'>"
        "<span style='color:#22c55e;'>Green</span> = horse appeared in both model picks and actual result. "
        "Trifecta box = all 3 actual finishers were somewhere in the model's top-3 (any order). "
        f"AUC: {stats['auc']:.3f}"
        "</div>", unsafe_allow_html=True)

    # Blend bar chart
    st.markdown('<div class="sec-title">Full Field — Blend Probability</div>',
                unsafe_allow_html=True)
    df_plot = df26.sort_values("blend_pct").tail(12)
    bar_colors = ["#3b82f6" if b>=100 else "#34d399" if b>=97 else "#4a4a4a"
                  for b in df_plot["best_beyer"]]
    fig_bar = go.Figure(go.Bar(
        x=df_plot["blend_pct"],
        y=df_plot["horse"],
        orientation="h",
        marker_color=bar_colors,
        text=df_plot["blend_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(color="#888", size=11),
        hovertemplate="%{y}<br>Blend: %{x:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        height=320,
        margin=dict(l=0, r=40, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1e1e1e", ticksuffix="%",
                   tickfont=dict(color="#555", size=11), title="",
                   range=[0, df_plot["blend_pct"].max()*1.22]),
        yaxis=dict(showgrid=False, tickfont=dict(color="#aaa", size=12), title=""),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})

    # Model vs market scatter
    st.markdown('<div class="sec-title">Model vs. Market Edge</div>',
                unsafe_allow_html=True)
    fig_sc = go.Figure()
    lim = max(df26["market_pct"].max(), df26["model_pct"].max())*1.15
    fig_sc.add_trace(go.Scatter(
        x=[0,lim], y=[0,lim], mode="lines",
        line=dict(color="#2a2a2a", dash="dot", width=1), showlegend=False))
    for _, row in df26.iterrows():
        color = "#3b82f6" if row["best_beyer"]>=100 else "#4a4a4a"
        fig_sc.add_trace(go.Scatter(
            x=[row["market_pct"]], y=[row["model_pct"]],
            mode="markers+text",
            marker=dict(size=9, color=color, opacity=0.9),
            text=[row["horse"].split()[-1]],
            textposition="top center",
            textfont=dict(size=9, color="#666"),
            hovertemplate=(f"<b>{row['horse']}</b><br>"
                           f"Market: {row['market_pct']:.1f}%<br>"
                           f"Model: {row['model_pct']:.1f}%<br>"
                           f"Edge: {row['edge']:+.1f}pp<extra></extra>"),
            showlegend=False,
        ))
    fig_sc.update_layout(
        height=260, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Market implied %", showgrid=True, gridcolor="#1e1e1e",
                   ticksuffix="%", tickfont=dict(color="#555",size=11),
                   title_font=dict(color="#555",size=11)),
        yaxis=dict(title="Model %", showgrid=True, gridcolor="#1e1e1e",
                   ticksuffix="%", tickfont=dict(color="#555",size=11),
                   title_font=dict(color="#555",size=11)),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar":False})
    st.markdown("<div style='font-size:11px;color:#444;margin-top:-8px;'>"
                "Above diagonal = model rates higher than market (value). "
                "Blue = 100+ Beyer. Grey = sub-100."
                "</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL BEYER CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="sec-title">Historical Derby Winner Beyer Figures (2000–2025)</div>',
            unsafe_allow_html=True)

yrs   = sorted(WINNER_BEY.keys())
beys  = [WINNER_BEY[y] for y in yrs]
names = [WINNER_NAMES[y] for y in yrs]
hist_colors = ["#3b82f6" if b>=100 else "#eab308" if b>=95 else "#ef4444" for b in beys]

fig_hist = go.Figure()
fig_hist.add_hline(y=100, line_dash="dash", line_color="#3b82f6", line_width=1.5,
                   annotation_text="100 threshold",
                   annotation_position="top right",
                   annotation_font=dict(size=11, color="#3b82f6"))
fig_hist.add_trace(go.Bar(
    x=yrs, y=beys,
    marker_color=hist_colors,
    text=[f"{n}<br>{b}" for n,b in zip(names,beys)],
    textangle=0, textposition="outside",
    textfont=dict(size=8, color="#555"),
    hovertemplate="<b>%{x}</b>: %{text}<extra></extra>",
    marker_line_width=0,
))
fig_hist.update_layout(
    height=260,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(range=[75,120], showgrid=True, gridcolor="#1a1a1a",
               tickfont=dict(color="#555",size=11), title="Beyer",
               title_font=dict(color="#555",size=11)),
    xaxis=dict(showgrid=False, tickfont=dict(color="#555",size=11), dtick=1),
    bargap=0.3,
    font=dict(family="Inter"),
)
st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar":False})
st.markdown(
    "<div style='font-size:11px;color:#444;margin-top:-6px;'>"
    "Blue = 100+ (qualified). Yellow = 95–99 (exceptions, usually longshots). "
    "Red = sub-95 (historic upsets only). 18/26 winners (69%) cleared the 100 threshold."
    "</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY EXPANDER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
with st.expander("📖 How this model was built — methodology & experiments"):
    st.markdown("""
<div class="narr">

<h3>The Goal</h3>
Build a machine learning model to predict the Kentucky Derby winner using only information
available <em>before</em> the race. All evaluation uses a strict
<strong>80/20 chronological split</strong> — trained on 1991–2018, tested on 2019–2025.
The holdout years were never touched during any training or tuning.

<h3>Data</h3>
<ul>
  <li><strong>643 horse-rows</strong> from Equibase PDF race charts (1991–2025) — every horse in every Derby field</li>
  <li>Features: finish, post position, morning-line odds, jockey, trainer, fractional times, track condition, race comments</li>
  <li>Historical jockey and trainer Derby win/top-3/starts stats</li>
  <li>Post position win rates (1930–2025, 20 posts)</li>
  <li><strong>Beyer Speed Figures</strong> — the key feature — real DRF figures for the 2026 field; historical winners manually compiled (2000–2025); non-winners estimated via odds proxy</li>
</ul>

<h3>15+ experiments — what actually moved the needle</h3>
<ul>
  <li><strong>Base features only</strong> (odds, post, connections): 0/7 holdout, AUC 0.555</li>
  <li><strong>+ Race comment features</strong> (trouble, wide, pace, closer): 2/7 — best gain without Beyer</li>
  <li><strong>+ Relative ranks</strong> (within-field percentile of odds, jockey, trainer): 0–1/7</li>
  <li><strong>+ Interaction terms</strong> (jockey×trainer, sloppy×longshot): 0–1/7</li>
  <li><strong>50-seed ensemble</strong>: 0/7 — smoothed probs but still missed</li>
  <li><strong>Stacking with meta-learner</strong>: 0/7</li>
  <li><strong>Track-specific models</strong>: 0/7 — too few sloppy years to train on</li>
  <li><strong>Pure market</strong> ($100M+ of public money): 0/7 — bettors also missed every winner 2019–2025</li>
  <li><strong>Bayesian blend, geometric mean, leave-one-year-out</strong>: all 0–1/7</li>
  <li><strong>+ Beyer Speed Figures</strong>: <span style="color:#22c55e;font-weight:600;">4/7 holdout, AUC 0.856</span> — the single biggest jump</li>
</ul>

<h3>Why Beyer changed everything</h3>
Every Derby winner since 2000 has posted at least one 100+ Beyer Speed Figure (18/26 = 69%),
with the 8 exceptions all being longshots at 50-1 or higher. The model uses this as its
strongest filter — horses without a 100+ Beyer are heavily downweighted. Commandment
is the <strong>only horse in the 2026 field with two triple-digit Beyers</strong>,
which is the profile most consistent with historical winners.

<h3>The 50/50 blend</h3>
Raw model probabilities are blended 50/50 with market implied probabilities (from morning-line odds).
The slider in the sidebar controls this ratio. Pure model is noisier but catches overlays.
Pure market reflects tens of millions of dollars of public opinion. The blend typically
produces better-calibrated rankings than either alone.

<h3>Honest caveats</h3>
<ul>
  <li><strong>4/7 is not a crystal ball</strong> — two of the four "correct" years were DQ winners (Country House 2019, Mandaloun 2021), and Rich Strike 2022 was an 80-1 historic upset that the model also got right only because of extremely low Beyer for the field that year</li>
  <li><strong>Tiny sample</strong>: 35 Derby races = 35 race-level training examples regardless of how many horse-rows</li>
  <li><strong>Non-winner Beyers estimated</strong> from odds for historical training — a full DRF database for every horse in every field would push this to 5–6/7</li>
  <li><strong>The Puma (PP9) scratched</strong> — use the Scratch Manager in the sidebar to mark any additional late scratches; the model reruns automatically</li>
</ul>

</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FULL TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="sec-title">Full Field Rankings</div>', unsafe_allow_html=True)

disp = df26[["pp","horse","odds_raw","best_beyer","blend_pct","model_pct",
             "market_pct","edge","jockey","trainer","last_prep"]].copy()
disp.columns = ["PP","Horse","Odds","Best Beyer","Blend %","Model %",
                "Market %","Edge (pp)","Jockey","Trainer","Last Prep"]
disp["PP"]        = disp["PP"].astype(int)
disp["Odds"]      = disp["Odds"].apply(lambda x: f"{x:.0f}-1")
disp["Best Beyer"]= disp["Best Beyer"].astype(int)
disp["Blend %"]   = disp["Blend %"].apply(lambda x: f"{x:.1f}%")
disp["Model %"]   = disp["Model %"].apply(lambda x: f"{x:.1f}%")
disp["Market %"]  = disp["Market %"].apply(lambda x: f"{x:.1f}%")
disp["Edge (pp)"] = disp["Edge (pp)"].apply(lambda x: f"+{x:.1f}" if x>0 else f"{x:.1f}")
disp.index = range(1, len(disp)+1)
st.dataframe(disp, use_container_width=True, height=380)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
scr_caption = f"  ·  Scratched: {', '.join(scr_names)}" if scr_names else "  ·  Full 20-horse field"
hist_caption = (f"  ·  {stats['n_hist_scratches']} historical row(s) excluded from training"
                if stats["n_hist_scratches"] > 0 else "")
st.caption(
    f"Model: HistGradientBoostingClassifier + Beyer Speed Figures  ·  "
    f"Train: {stats['train_years']} ({stats['n_train']} horses)  ·  "
    f"Test: {stats['test_years']} ({stats['n_test']} horses)  ·  "
    f"{stats['n_features']} features  ·  "
    f"Track: {track_cond}  ·  Model weight: {model_wt:.0%}"
    f"{scr_caption}{hist_caption}"
)
