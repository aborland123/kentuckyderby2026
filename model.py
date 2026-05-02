# """
# Kentucky Derby 2026 Pre-Race Prediction Model
# Run with: python3 derby_2026_model.py
# Requires: pandas, numpy, scikit-learn
#   pip install pandas numpy scikit-learn
# """
 
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# import warnings
# warnings.filterwarnings('ignore')
 
# # ── PATHS — update these if your files are elsewhere ──────────────────────────
# MASTER_CSV   = "kentucky_derby_equibase_MASTER.csv"
# JOCKEY_CSV   = "derby_jockey_stats.csv"
# TRAINER_CSV  = "derby_trainer_stats.csv"
# PP_CSV       = "derby_post_positions.csv"
 
# # ── 2026 FIELD — edit odds/pp if they change ──────────────────────────────────
# field_2026 = [
#     {"pp":  1, "horse": "Renegade",        "jockey": "Ortiz, Irad",       "trainer": "Pletcher, Todd",  "odds":  4.0, "record": "5-2-2-1"},
#     {"pp":  2, "horse": "Albus",           "jockey": "Franco, Manny",     "trainer": "Mott, Riley",     "odds": 30.0, "record": "4-2-0-1"},
#     {"pp":  3, "horse": "Intrepido",       "jockey": "Berrios, Hector",   "trainer": "Mullins, Jeff",   "odds": 50.0, "record": "6-2-1-0"},
#     {"pp":  4, "horse": "Litmus Test",     "jockey": "Garcia, Martin",    "trainer": "Baffert, Bob",    "odds": 30.0, "record": "7-2-0-2"},
#     {"pp":  5, "horse": "Right to Party",  "jockey": "Elliott, Chris",    "trainer": "McPeek, Kenneth", "odds": 30.0, "record": "4-1-1-2"},
#     {"pp":  6, "horse": "Commandment",     "jockey": "Saez, Luis",        "trainer": "Cox, Brad",       "odds":  6.0, "record": "5-4-0-0"},
#     {"pp":  7, "horse": "Danon Bourbon",   "jockey": "Nishimura, Atsuya", "trainer": "Ikezoe, Manabu",  "odds": 20.0, "record": "3-3-0-0"},
#     {"pp":  8, "horse": "So Happy",        "jockey": "Smith, Mike",       "trainer": "Glatt, Mark",     "odds": 15.0, "record": "4-3-0-1"},
#     {"pp":  9, "horse": "The Puma",        "jockey": "Castellano, Javier","trainer": "Delgado, Gustavo","odds": 10.0, "record": "4-1-2-1"},
#     {"pp": 10, "horse": "Wonder Dean",     "jockey": "Demuro, Cristian",  "trainer": "Takayanagi, D.",  "odds": 30.0, "record": "6-2-2-0"},
#     {"pp": 11, "horse": "Incredibolt",     "jockey": "Torres, Jaime",     "trainer": "Mott, Riley",     "odds": 20.0, "record": "5-3-0-0"},
#     {"pp": 12, "horse": "Chief Wallabe",   "jockey": "Alvarado, Junior",  "trainer": "Mott, William",   "odds":  8.0, "record": "3-1-1-1"},
#     {"pp": 13, "horse": "Silent Tactic",   "jockey": "Torres, Cristian",  "trainer": "Casse, Mark",     "odds": 20.0, "record": "6-2-4-0"},
#     {"pp": 14, "horse": "Potente",         "jockey": "Hernandez, Juan",   "trainer": "Baffert, Bob",    "odds": 20.0, "record": "3-2-1-0"},
#     {"pp": 15, "horse": "Emerging Market", "jockey": "Prat, Flavien",     "trainer": "Brown, Chad",     "odds": 15.0, "record": "2-2-0-0"},
#     {"pp": 16, "horse": "Pavlovian",       "jockey": "Maldonado, Edwin",  "trainer": "O'Neill, Doug",   "odds": 30.0, "record": "10-2-4-1"},
#     {"pp": 17, "horse": "Six Speed",       "jockey": "Hernandez, Brian",  "trainer": "Seemar, Bhupat",  "odds": 50.0, "record": "5-3-1-1"},
#     {"pp": 18, "horse": "Further Ado",     "jockey": "Velazquez, John",   "trainer": "Cox, Brad",       "odds":  6.0, "record": "6-3-1-1"},
#     {"pp": 19, "horse": "Golden Tempo",    "jockey": "Ortiz, Jose",       "trainer": "DeVaux, Cherie",  "odds": 30.0, "record": "4-2-0-2"},
#     {"pp": 20, "horse": "Fulleffort",      "jockey": "Gaffalione, Tyler", "trainer": "Cox, Brad",       "odds": 20.0, "record": "7-3-2-1"},
# ]
 
# # ── LOAD HISTORICAL DATA ──────────────────────────────────────────────────────
# print("Loading historical data...")
# master  = pd.read_csv(MASTER_CSV)
# jockey  = pd.read_csv(JOCKEY_CSV)
# trainer = pd.read_csv(TRAINER_CSV)
# pp_stats= pd.read_csv(PP_CSV)
 
# master['won']  = (master['finish'] == 1).astype(int)
# master['top3'] = (master['finish'] <= 3).astype(int)
# master['year'] = master['year'].astype(int)
# master['pp']   = pd.to_numeric(master['pp'],   errors='coerce')
# master['odds'] = pd.to_numeric(master['odds'], errors='coerce')
 
# # ── FEATURE ENGINEERING HELPERS ──────────────────────────────────────────────
# pp_lk = pp_stats.set_index('post')[['win_pct','top3_pct']].to_dict('index')
 
# def pp_feat(pp, stat):
#     if pd.isna(pp): return 5.0 if stat=='win_pct' else 20.0
#     return pp_lk.get(int(pp), {}).get(stat, 5.0 if stat=='win_pct' else 20.0)
 
# j_lk = jockey.set_index('jockey')[['win_pct','top3_pct','derby_starts']].to_dict('index')
# t_lk = trainer.set_index('trainer')[['win_pct','top3_pct','derby_starts']].to_dict('index')
 
# def lookup(lk, name, stat, default):
#     if not name: return default
#     name_l = str(name).lower()
#     for k, v in lk.items():
#         k_l = str(k).lower()
#         if k_l in name_l or name_l in k_l:
#             return v[stat]
#     return default
 
# # ── HISTORICAL FEATURES (PRE-RACE ONLY) ───────────────────────────────────────
# master['pp_win']    = master['pp'].apply(lambda x: pp_feat(x,'win_pct'))
# master['pp_t3']     = master['pp'].apply(lambda x: pp_feat(x,'top3_pct'))
# master['pp_cursed'] = (master['pp'] == 17).astype(int)
 
# master['track_fast']   = master['track_condition'].str.contains('Fast',   na=False).astype(int)
# master['track_sloppy'] = master['track_condition'].str.contains('Sloppy', na=False).astype(int)
 
# master['j_win']    = master['jockey'].apply(lambda x: lookup(j_lk, x, 'win_pct',    5.0))
# master['j_t3']     = master['jockey'].apply(lambda x: lookup(j_lk, x, 'top3_pct',  20.0))
# master['j_starts'] = master['jockey'].apply(lambda x: lookup(j_lk, x, 'derby_starts', 1))
# master['t_win']    = master['trainer'].apply(lambda x: lookup(t_lk, x, 'win_pct',    3.0))
# master['t_t3']     = master['trainer'].apply(lambda x: lookup(t_lk, x, 'top3_pct',  15.0))
# master['t_starts'] = master['trainer'].apply(lambda x: lookup(t_lk, x, 'derby_starts', 1))
 
# master['impl_prob'] = 1 / (master['odds'] + 1)
# master['is_fav']    = (master['odds'] <  5.0).astype(int)
# master['is_shot']   = (master['odds'] > 25.0).astype(int)
 
# # Win record parsing (wins / total starts)
# def parse_record(rec):
#     """Parse 'starts-wins-places-shows' or return defaults."""
#     try:
#         parts = str(rec).split('-')
#         starts = int(parts[0]); wins = int(parts[1])
#         return starts, wins, wins / max(starts, 1)
#     except:
#         return 5, 1, 0.2
 
# fs = master.groupby('year')['horse'].count().reset_index()
# fs.columns = ['year','field_size']
# master = master.merge(fs, on='year', how='left')
 
# # ── PRE-RACE FEATURE LIST ─────────────────────────────────────────────────────
# FEATURES = [
#     'pp', 'pp_win', 'pp_t3', 'pp_cursed',
#     'track_fast', 'track_sloppy',
#     'j_win', 'j_t3', 'j_starts',
#     't_win', 't_t3', 't_starts',
#     'impl_prob', 'is_fav', 'is_shot',
#     'field_size',
# ]
 
# df = master[FEATURES + ['won','top3','year']].copy()
# df = df[df['impl_prob'].notna()].reset_index(drop=True)
 
# X  = df[FEATURES].astype(float)
# y  = df['won']
# yt = df['top3']
 
# # ── TRAIN MODELS ──────────────────────────────────────────────────────────────
# print("Training win model...")
# m_win = HistGradientBoostingClassifier(
#     max_iter=500, max_depth=4, learning_rate=0.03,
#     min_samples_leaf=8, random_state=42, class_weight='balanced'
# )
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# win_auc = cross_val_score(m_win, X, y, cv=cv, scoring='roc_auc')
# m_win.fit(X, y)
# print(f"  Win model  CV AUC: {win_auc.mean():.3f} ± {win_auc.std():.3f}")
 
# print("Training top-3 model...")
# m_top3 = HistGradientBoostingClassifier(
#     max_iter=500, max_depth=4, learning_rate=0.03,
#     min_samples_leaf=8, random_state=42, class_weight='balanced'
# )
# t3_auc = cross_val_score(m_top3, X, yt, cv=cv, scoring='roc_auc')
# m_top3.fit(X, yt)
# print(f"  Top3 model CV AUC: {t3_auc.mean():.3f} ± {t3_auc.std():.3f}")
 
# # ── HISTORICAL YEAR-BY-YEAR CHECK ─────────────────────────────────────────────
# print("\nHistorical predictions (highest win_prob vs actual winner):")
# df2 = df.copy()
# df2['horse']    = master.loc[df.index, 'horse'].values
# df2['win_prob'] = m_win.predict_proba(X)[:,1]
# correct = 0
# for yr in sorted(df2['year'].unique()):
#     yrdf = df2[df2['year']==yr].sort_values('win_prob', ascending=False)
#     pred   = yrdf.iloc[0]['horse']
#     actual = yrdf[yrdf['won']==1]['horse'].values
#     actual_name = actual[0] if len(actual) > 0 else '?'
#     hit = pred == actual_name
#     if hit: correct += 1
#     symbol = '✓' if hit else '✗'
#     print(f"  {yr}: {symbol}  pred={pred:<25s}  actual={actual_name}")
# total = len(df2['year'].unique())
# print(f"\n  Pre-race model: {correct}/{total} = {correct/total:.1%}")
 
# # ── SCORE 2026 FIELD ──────────────────────────────────────────────────────────
# print("\n" + "="*65)
# print("2026 KENTUCKY DERBY — PRE-RACE MODEL SCORES")
# print("="*65)
 
# # Track assumption: if unknown, default to Fast (most common)
# TRACK_2026 = "Fast"   # change to "Sloppy" if rain in forecast
 
# field_size_2026 = len(field_2026)
 
# rows_2026 = []
# for h in field_2026:
#     # parse win record
#     starts, wins, win_rate = parse_record(h['record'])
 
#     row = {
#         'pp':         h['pp'],
#         'pp_win':     pp_feat(h['pp'], 'win_pct'),
#         'pp_t3':      pp_feat(h['pp'], 'top3_pct'),
#         'pp_cursed':  int(h['pp'] == 17),
#         'track_fast':   int(TRACK_2026 == 'Fast'),
#         'track_sloppy': int(TRACK_2026 == 'Sloppy'),
#         'j_win':    lookup(j_lk, h['jockey'],  'win_pct',    5.0),
#         'j_t3':     lookup(j_lk, h['jockey'],  'top3_pct',  20.0),
#         'j_starts': lookup(j_lk, h['jockey'],  'derby_starts', 1),
#         't_win':    lookup(t_lk, h['trainer'], 'win_pct',    3.0),
#         't_t3':     lookup(t_lk, h['trainer'], 'top3_pct',  15.0),
#         't_starts': lookup(t_lk, h['trainer'], 'derby_starts', 1),
#         'impl_prob': 1 / (h['odds'] + 1),
#         'is_fav':    int(h['odds'] < 5.0),
#         'is_shot':   int(h['odds'] > 25.0),
#         'field_size': field_size_2026,
#     }
#     rows_2026.append(row)
 
# X_2026 = pd.DataFrame(rows_2026)[FEATURES].astype(float)
# win_probs  = m_win.predict_proba(X_2026)[:,1]
# top3_probs = m_top3.predict_proba(X_2026)[:,1]
 
# # Normalize win probs to sum to ~1 (like a proper market)
# win_probs_norm = win_probs / win_probs.sum()
 
# results = []
# for i, h in enumerate(field_2026):
#     results.append({
#         'rank':      0,
#         'pp':        h['pp'],
#         'horse':     h['horse'],
#         'jockey':    h['jockey'],
#         'trainer':   h['trainer'],
#         'odds':      h['odds'],
#         'record':    h['record'],
#         'win_prob':  round(win_probs_norm[i] * 100, 1),
#         'top3_prob': round(top3_probs[i] * 100, 1),
#         'raw_win':   round(win_probs[i] * 1000, 2),   # useful for relative comparison
#         # key driver breakdown
#         'pp_score':       round(pp_feat(h['pp'],'win_pct'), 1),
#         'j_win_pct':      round(lookup(j_lk, h['jockey'],  'win_pct', 5.0), 1),
#         't_win_pct':      round(lookup(t_lk, h['trainer'], 'win_pct', 3.0), 1),
#     })
 
# results.sort(key=lambda x: x['win_prob'], reverse=True)
# for i, r in enumerate(results):
#     r['rank'] = i + 1
 
# print(f"\n{'Rank':>4}  {'PP':>3}  {'Horse':<17}  {'Odds':>6}  {'Win%':>5}  {'Top3%':>6}  {'J-Win%':>7}  {'T-Win%':>7}  Trainer")
# print("-"*100)
# for r in results:
#     print(f"  {r['rank']:>2}.  #{r['pp']:<3}  {r['horse']:<17}  {r['odds']:>5.0f}-1  {r['win_prob']:>5.1f}%  {r['top3_prob']:>5.1f}%  {r['j_win_pct']:>6.1f}%  {r['t_win_pct']:>6.1f}%  {r['trainer']}")
 
# print("\n" + "="*65)
# print("MODEL PICKS")
# print("="*65)
# top3 = results[:3]
# print(f"  WIN:    #{top3[0]['pp']} {top3[0]['horse']} ({top3[0]['odds']:.0f}-1)  —  model win prob {top3[0]['win_prob']:.1f}%")
# print(f"  PLACE:  #{top3[1]['pp']} {top3[1]['horse']} ({top3[1]['odds']:.0f}-1)  —  model win prob {top3[1]['win_prob']:.1f}%")
# print(f"  SHOW:   #{top3[2]['pp']} {top3[2]['horse']} ({top3[2]['odds']:.0f}-1)  —  model win prob {top3[2]['win_prob']:.1f}%")
 
# print(f"""
# NOTES:
#   Track assumption : {TRACK_2026} — change TRACK_2026 at top of file if weather changes
#   Model features   : post position history, jockey/trainer Derby records, morning-line odds
#   NOT included     : past performance speed figures, workout times, distance history
#   Honest AUC       : {win_auc.mean():.3f} (pre-race only — no in-race position data)
  
#   To change odds (e.g. if ML shifts), edit the 'odds' values in field_2026 and re-run.
#   To run a sloppy-track scenario, change TRACK_2026 = "Sloppy"
# """)
 
# # Save JSON for dashboard
# import json
# out = {
#     "year": 2026,
#     "track": TRACK_2026,
#     "win_auc": round(win_auc.mean(), 3),
#     "top3_auc": round(t3_auc.mean(), 3),
#     "results": results
# }
# with open("derby_2026_predictions.json","w") as f:
#     json.dump(out, f, indent=2)
# print("Saved: derby_2026_predictions.json")

# /////////


"""
Kentucky Derby 2026 — Pre-Race Prediction Model  (v2)
──────────────────────────────────────────────────────
Honest 80/20 chronological train/test split:
  Train : 1991–2018  (28 years, ~514 horses)
  Test  : 2019–2025  (7 years,  ~129 horses)
  Predict: 2026 field
 
Run:  python3 derby_2026_model_v2.py
Deps: pip install pandas numpy scikit-learn
"""
 
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
 
# ── PATHS ──────────────────────────────────────────────────────────────────────
MASTER_CSV   = "kentucky_derby_equibase_MASTER.csv"
JOCKEY_CSV   = "derby_jockey_stats.csv"
TRAINER_CSV  = "derby_trainer_stats.csv"
PP_CSV       = "derby_post_positions.csv"
 
TRACK_2026   = "Fast"   # flip to "Sloppy" if rain forecast
 
# ── 2026 FIELD ─────────────────────────────────────────────────────────────────
FIELD_2026 = [
    {"pp":  1, "horse": "Renegade",        "jockey": "Ortiz, Irad",        "trainer": "Pletcher, Todd",   "odds":  4.0, "record": "5-2-2-1"},
    {"pp":  2, "horse": "Albus",           "jockey": "Franco, Manny",      "trainer": "Mott, Riley",      "odds": 30.0, "record": "4-2-0-1"},
    {"pp":  3, "horse": "Intrepido",       "jockey": "Berrios, Hector",    "trainer": "Mullins, Jeff",    "odds": 50.0, "record": "6-2-1-0"},
    {"pp":  4, "horse": "Litmus Test",     "jockey": "Garcia, Martin",     "trainer": "Baffert, Bob",     "odds": 30.0, "record": "7-2-0-2"},
    {"pp":  5, "horse": "Right to Party",  "jockey": "Elliott, Chris",     "trainer": "McPeek, Kenneth",  "odds": 30.0, "record": "4-1-1-2"},
    {"pp":  6, "horse": "Commandment",     "jockey": "Saez, Luis",         "trainer": "Cox, Brad",        "odds":  6.0, "record": "5-4-0-0"},
    {"pp":  7, "horse": "Danon Bourbon",   "jockey": "Nishimura, Atsuya",  "trainer": "Ikezoe, Manabu",   "odds": 20.0, "record": "3-3-0-0"},
    {"pp":  8, "horse": "So Happy",        "jockey": "Smith, Mike",        "trainer": "Glatt, Mark",      "odds": 15.0, "record": "4-3-0-1"},
    {"pp":  9, "horse": "The Puma",        "jockey": "Castellano, Javier", "trainer": "Delgado, Gustavo", "odds": 10.0, "record": "4-1-2-1"},
    {"pp": 10, "horse": "Wonder Dean",     "jockey": "Demuro, Cristian",   "trainer": "Takayanagi, D.",   "odds": 30.0, "record": "6-2-2-0"},
    {"pp": 11, "horse": "Incredibolt",     "jockey": "Torres, Jaime",      "trainer": "Mott, Riley",      "odds": 20.0, "record": "5-3-0-0"},
    {"pp": 12, "horse": "Chief Wallabe",   "jockey": "Alvarado, Junior",   "trainer": "Mott, William",    "odds":  8.0, "record": "3-1-1-1"},
    {"pp": 13, "horse": "Silent Tactic",   "jockey": "Torres, Cristian",   "trainer": "Casse, Mark",      "odds": 20.0, "record": "6-2-4-0"},
    {"pp": 14, "horse": "Potente",         "jockey": "Hernandez, Juan",    "trainer": "Baffert, Bob",     "odds": 20.0, "record": "3-2-1-0"},
    {"pp": 15, "horse": "Emerging Market", "jockey": "Prat, Flavien",      "trainer": "Brown, Chad",      "odds": 15.0, "record": "2-2-0-0"},
    {"pp": 16, "horse": "Pavlovian",       "jockey": "Maldonado, Edwin",   "trainer": "O'Neill, Doug",    "odds": 30.0, "record": "10-2-4-1"},
    {"pp": 17, "horse": "Six Speed",       "jockey": "Hernandez, Brian",   "trainer": "Seemar, Bhupat",   "odds": 50.0, "record": "5-3-1-1"},
    {"pp": 18, "horse": "Further Ado",     "jockey": "Velazquez, John",    "trainer": "Cox, Brad",        "odds":  6.0, "record": "6-3-1-1"},
    {"pp": 19, "horse": "Golden Tempo",    "jockey": "Ortiz, Jose",        "trainer": "DeVaux, Cherie",   "odds": 30.0, "record": "4-2-0-2"},
    {"pp": 20, "horse": "Fulleffort",      "jockey": "Gaffalione, Tyler",  "trainer": "Cox, Brad",        "odds": 20.0, "record": "7-3-2-1"},
]
 
# ── LOAD DATA ──────────────────────────────────────────────────────────────────
print("Loading data...")
master   = pd.read_csv(MASTER_CSV)
jockey   = pd.read_csv(JOCKEY_CSV)
trainer  = pd.read_csv(TRAINER_CSV)
pp_stats = pd.read_csv(PP_CSV)
 
master['won']  = (master['finish'] == 1).astype(int)
master['top3'] = (master['finish'] <= 3).astype(int)
master['year'] = master['year'].astype(int)
master['pp']   = pd.to_numeric(master['pp'],   errors='coerce')
master['odds'] = pd.to_numeric(master['odds'], errors='coerce')
 
# ── LOOKUP HELPERS ─────────────────────────────────────────────────────────────
pp_lk = pp_stats.set_index('post')[['win_pct','top3_pct']].to_dict('index')
j_lk  = jockey.set_index('jockey')[['win_pct','top3_pct','derby_starts']].to_dict('index')
t_lk  = trainer.set_index('trainer')[['win_pct','top3_pct','derby_starts']].to_dict('index')
 
def pp_f(pp, stat):
    if pd.isna(pp): return 5.0 if stat == 'win_pct' else 20.0
    return pp_lk.get(int(pp), {}).get(stat, 5.0 if stat == 'win_pct' else 20.0)
 
def lkup(lk, name, stat, default):
    if not name: return default
    nl = str(name).lower()
    for k, v in lk.items():
        if str(k).lower() in nl or nl in str(k).lower():
            return v[stat]
    return default
 
# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
master['pp_win']    = master['pp'].apply(lambda x: pp_f(x, 'win_pct'))
master['pp_t3']     = master['pp'].apply(lambda x: pp_f(x, 'top3_pct'))
master['pp_cursed'] = (master['pp'] == 17).astype(int)
master['track_fast']   = master['track_condition'].str.contains('Fast',   na=False).astype(int)
master['track_sloppy'] = master['track_condition'].str.contains('Sloppy', na=False).astype(int)
master['j_win']    = master['jockey'].apply(lambda x: lkup(j_lk, x, 'win_pct',      5.0))
master['j_t3']     = master['jockey'].apply(lambda x: lkup(j_lk, x, 'top3_pct',    20.0))
master['j_starts'] = master['jockey'].apply(lambda x: lkup(j_lk, x, 'derby_starts',   1))
master['t_win']    = master['trainer'].apply(lambda x: lkup(t_lk, x, 'win_pct',     3.0))
master['t_t3']     = master['trainer'].apply(lambda x: lkup(t_lk, x, 'top3_pct',   15.0))
master['t_starts'] = master['trainer'].apply(lambda x: lkup(t_lk, x, 'derby_starts',  1))
master['impl_prob'] = 1 / (master['odds'] + 1)
master['is_fav']    = (master['odds'] <  5.0).astype(int)
master['is_shot']   = (master['odds'] > 25.0).astype(int)
fs = master.groupby('year')['horse'].count().reset_index()
fs.columns = ['year', 'field_size']
master = master.merge(fs, on='year', how='left')
 
FEATS = [
    'pp', 'pp_win', 'pp_t3', 'pp_cursed',
    'track_fast', 'track_sloppy',
    'j_win', 'j_t3', 'j_starts',
    't_win', 't_t3', 't_starts',
    'impl_prob', 'is_fav', 'is_shot',
    'field_size',
]
 
df = master[FEATS + ['won', 'top3', 'year', 'horse']].copy()
df = df[df['impl_prob'].notna()].reset_index(drop=True)
 
# ── CHRONOLOGICAL 80/20 SPLIT ──────────────────────────────────────────────────
all_years   = sorted(df['year'].unique())
n_train     = int(len(all_years) * 0.8)
train_years = all_years[:n_train]   # 1991–2018
test_years  = all_years[n_train:]   # 2019–2025
 
train = df[df['year'].isin(train_years)].reset_index(drop=True)
test  = df[df['year'].isin(test_years)].reset_index(drop=True)
 
X_tr = train[FEATS].astype(float);  y_tr = train['won'];  yt_tr = train['top3']
X_te = test[FEATS].astype(float);   y_te = test['won'];   yt_te = test['top3']
 
print(f"Train: {train_years[0]}–{train_years[-1]}  ({len(train_years)} yrs, {len(train)} horses)")
print(f"Test : {test_years[0]}–{test_years[-1]}   ({len(test_years)} yrs, {len(test)} horses)")
 
# ── TRAIN ──────────────────────────────────────────────────────────────────────
print("\nTraining models...")
m_win = HistGradientBoostingClassifier(
    max_iter=500, max_depth=4, learning_rate=0.03,
    min_samples_leaf=8, random_state=42, class_weight='balanced')
m_win.fit(X_tr, y_tr)
 
m_top3 = HistGradientBoostingClassifier(
    max_iter=500, max_depth=4, learning_rate=0.03,
    min_samples_leaf=8, random_state=42, class_weight='balanced')
m_top3.fit(X_tr, yt_tr)
 
# ── EVALUATE ───────────────────────────────────────────────────────────────────
tr_auc  = roc_auc_score(y_tr,  m_win.predict_proba(X_tr)[:,1])
te_auc  = roc_auc_score(y_te,  m_win.predict_proba(X_te)[:,1])
tr_t3   = roc_auc_score(yt_tr, m_top3.predict_proba(X_tr)[:,1])
te_t3   = roc_auc_score(yt_te, m_top3.predict_proba(X_te)[:,1])
 
print(f"\n{'─'*50}")
print(f"  Win model  │ Train AUC: {tr_auc:.3f} │ Test AUC: {te_auc:.3f}")
print(f"  Top3 model │ Train AUC: {tr_t3:.3f} │ Test AUC: {te_t3:.3f}")
print(f"{'─'*50}")
print(f"  (Train=in-sample, Test=never-seen holdout)")
 
# ── HOLDOUT YEAR-BY-YEAR ────────────────────────────────────────────────────────
te2 = test.copy()
te2['win_prob'] = m_win.predict_proba(X_te)[:,1]
te2['t3_prob']  = m_top3.predict_proba(X_te)[:,1]
 
print(f"\nHoldout results — {test_years[0]}–{test_years[-1]}:")
print(f"  {'Year':>4}  {'':2}  {'Model pick':<25}  {'Actual winner':<25}  Winner's rank")
print(f"  {'─'*75}")
correct_te = 0
for yr in test_years:
    yrdf = te2[te2['year'] == yr].sort_values('win_prob', ascending=False).reset_index(drop=True)
    pred = yrdf.iloc[0]['horse']
    actual_rows = yrdf[yrdf['won'] == 1]['horse'].values
    actual = actual_rows[0] if len(actual_rows) > 0 else '?'
    hit = (pred == actual)
    if hit: correct_te += 1
    sym = '✓' if hit else '✗'
    # find where the actual winner ranked
    rank = yrdf[yrdf['horse'] == actual].index[0] + 1 if actual in yrdf['horse'].values else '?'
    print(f"  {yr}  {sym}  {pred:<25}  {actual:<25}  #{rank}")
 
print(f"\n  Holdout accuracy : {correct_te}/{len(test_years)} = {correct_te/len(test_years):.1%}")
print(f"  Honest test AUC  : {te_auc:.3f}  ← use this number, not the training AUC")
print(f"\n  Context: AUC 0.5 = coin flip. AUC 0.555 still beats random —")
print(f"  the model has mild signal on pre-race features alone.")
print(f"  (Most of the strong signal lives in in-race position data.)")
 
# ── SCORE 2026 FIELD ────────────────────────────────────────────────────────────
# Retrain on ALL historical data (1991–2025) before predicting 2026
print(f"\n{'─'*50}")
print(f"Retraining on full 1991–2025 data before scoring 2026...")
X_all = df[FEATS].astype(float)
y_all = df['won'];  yt_all = df['top3']
m_win.fit(X_all, y_all)
m_top3.fit(X_all, yt_all)
 
def build_row(h, track, n_field):
    return {
        'pp':         h['pp'],
        'pp_win':     pp_f(h['pp'], 'win_pct'),
        'pp_t3':      pp_f(h['pp'], 'top3_pct'),
        'pp_cursed':  int(h['pp'] == 17),
        'track_fast':   int(track == 'Fast'),
        'track_sloppy': int(track == 'Sloppy'),
        'j_win':    lkup(j_lk, h['jockey'],  'win_pct',      5.0),
        'j_t3':     lkup(j_lk, h['jockey'],  'top3_pct',    20.0),
        'j_starts': lkup(j_lk, h['jockey'],  'derby_starts',   1),
        't_win':    lkup(t_lk, h['trainer'], 'win_pct',      3.0),
        't_t3':     lkup(t_lk, h['trainer'], 'top3_pct',    15.0),
        't_starts': lkup(t_lk, h['trainer'], 'derby_starts',   1),
        'impl_prob': 1 / (h['odds'] + 1),
        'is_fav':    int(h['odds'] <  5.0),
        'is_shot':   int(h['odds'] > 25.0),
        'field_size': n_field,
    }
 
rows = [build_row(h, TRACK_2026, len(FIELD_2026)) for h in FIELD_2026]
X26  = pd.DataFrame(rows)[FEATS].astype(float)
wp   = m_win.predict_proba(X26)[:,1]
tp   = m_top3.predict_proba(X26)[:,1]
wp_n = wp / wp.sum()   # normalise so field sums to ~100%
 
results = []
for i, h in enumerate(FIELD_2026):
    edge = round((wp_n[i] - (1/(h['odds']+1)))*100, 1)
    results.append({
        'pp':       h['pp'],
        'horse':    h['horse'],
        'jockey':   h['jockey'],
        'trainer':  h['trainer'],
        'odds':     h['odds'],
        'record':   h['record'],
        'win_pct':  round(wp_n[i]*100, 1),
        'top3_pct': round(tp[i]*100, 1),
        'edge':     edge,
        'j_derby_win': round(lkup(j_lk, h['jockey'], 'win_pct', 5.0), 1),
        't_derby_win': round(lkup(t_lk, h['trainer'], 'win_pct', 3.0), 1),
        'pp_hist_win': round(pp_f(h['pp'], 'win_pct'), 1),
    })
 
results.sort(key=lambda x: x['win_pct'], reverse=True)
 
print(f"\n{'='*75}")
print(f"  2026 KENTUCKY DERBY — MODEL PREDICTIONS  (track: {TRACK_2026})")
print(f"{'='*75}")
print(f"  {'Rank':>4}  {'PP':<4}  {'Horse':<17}  {'Odds':>6}  {'Win%':>5}  {'Top3%':>6}  {'Edge':>6}  Trainer")
print(f"  {'─'*73}")
for i, r in enumerate(results, 1):
    edge_str = f"+{r['edge']:.1f}" if r['edge'] > 0 else f"{r['edge']:.1f}"
    print(f"  {i:>4}.  #{r['pp']:<3}  {r['horse']:<17}  {r['odds']:>5.0f}-1  "
          f"{r['win_pct']:>5.1f}%  {r['top3_pct']:>5.1f}%  {edge_str:>6}pp  {r['trainer']}")
 
print(f"\n  Edge = model win% minus market implied% (positive = model likes more than market)")
 
print(f"\n{'='*75}")
print(f"  MODEL PICKS")
print(f"{'='*75}")
for label, idx in [("WIN  ", 0), ("PLACE", 1), ("SHOW ", 2)]:
    r = results[idx]
    print(f"  {label}  →  #{r['pp']} {r['horse']:<18}  {r['odds']:.0f}-1  "
          f"(model {r['win_pct']:.1f}% win / {r['top3_pct']:.1f}% top-3)")
 
# Value plays: model likes more than market by >2pp
value = [r for r in results if r['edge'] > 2.0]
if value:
    print(f"\n  Value plays (model win% beats market by >2pp):")
    for r in value:
        print(f"    #{r['pp']} {r['horse']:<18}  {r['odds']:.0f}-1  edge +{r['edge']:.1f}pp")
 
print(f"""
{'─'*75}
  HONEST NUMBERS
    Train AUC (1991–2018) : {tr_auc:.3f}  ← inflated, model saw this data
    Test  AUC (2019–2025) : {te_auc:.3f}  ← real estimate of predictive power
    Holdout picks correct  : {correct_te}/{len(test_years)} years
 
  WHAT THIS MODEL DOES AND DOESN'T KNOW
    ✓  Post position historical win rates (1930–2025)
    ✓  Morning-line odds (market consensus)
    ✓  Jockey Derby win/top-3 history
    ✓  Trainer Derby win/top-3 history
    ✓  Track condition (fast vs sloppy)
    ✗  Speed figures / Beyer numbers
    ✗  Workout times
    ✗  Distance/surface history
    ✗  Pace scenarios
    ✗  In-race position (only available after the race)
 
  To flip to sloppy track:  set TRACK_2026 = "Sloppy" at top of file
  To update odds on race morning: edit the odds values in FIELD_2026
{'─'*75}
""")
 
import json
out = {
    "year": 2026, "track": TRACK_2026,
    "train_years": f"{train_years[0]}-{train_years[-1]}",
    "test_years":  f"{test_years[0]}-{test_years[-1]}",
    "train_auc": round(tr_auc, 3), "test_auc": round(te_auc, 3),
    "holdout_correct": f"{correct_te}/{len(test_years)}",
    "results": results,
}
with open("derby_2026_predictions_v2.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved: derby_2026_predictions_v2.json")