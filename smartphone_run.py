"""
smartphone_run.py
──────────────────────────────────────────────────────────────────────────────
Top‑level entry point: builds the MILP via `smartphone_milp_model.build_model`,
completes *all* cost‑expression terms, runs optimisation, and writes a
fully‑compliant `plan_submission_template.db`.

USAGE
─────
python smartphone_run.py           # default: daily=True, threads=32
python smartphone_run.py --weekly  # faster weekly approx
python smartphone_run.py -t 16
"""
from __future__ import annotations
import argparse, datetime as dt, sqlite3, shutil, tempfile

import numpy as np
import gurobipy as gp

from smartphone_config_utils import (
    BASE_DIR, CONTAINER_CAP, TON_PENALTY_USD, BIG_M,
    MODES_FCWH,
    week_monday, ceil_div_expr
)
import smartphone_data_prep as dp
from smartphone_milp_model import build_model

# ═══════════════ CLI ═══════════════════════════════════════════════════════
P = argparse.ArgumentParser()
P.add_argument("--weekly", action="store_true",
               help="Use week‑level approximate model")
P.add_argument("-t","--threads", type=int, default=32)
args = P.parse_args()

mdl, V = build_model(daily=not args.weekly, threads=args.threads)

# ═══════════════ FILL COST EXPRESSIONS (production, wage, transport, CO₂) ═
print("⋆  Expanding linear cost expressions …")

prod_cost = V["cost_terms"]["prod"]
trans_cost= V["cost_terms"]["trans"]
co2_prod  = V["cost_terms"]["co2p"]
co2_tran  = V["cost_terms"]["co2t"]
change_pen= V["cost_terms"]["change"]

# Production & wage cost loop
for t, f, s in V["ProdR"].keys():
    iso = dp.iso_site[f]
    fx_date = t if isinstance(t, dt.date) else t.start_time.date()
    fx  = dp.FX_RATE[(fx_date, iso)]
    base = dp.prod_cost.loc[(dp.prod_cost.factory == f) &
                            (dp.prod_cost.sku == s),
                            "base_cost_usd"].iloc[0]
    wage = dp.lab_pol.loc[
        (dp.lab_pol.country == iso) &
        (dp.lab_pol.year == fx_date.year),
        "regular_wage_local"
    ].iloc[0] / fx
    otmul= dp.lab_pol.loc[dp.lab_pol.country == iso,
                          "ot_mult"].iloc[0]
    hrs  = dp.lab_req.loc[dp.lab_req.sku == s,
                          "labour_hours_per_unit"].iloc[0]
    date_val = t if isinstance(t, dt.date) else t.start_time.date()
    is_hol = date_val in dp.holiday.loc[dp.holiday.country == iso, "date"].dt.date.tolist()
    # regular
    prod_cost += base * V["ProdR"][t, f, s]
    prod_cost += wage * hrs * V["ProdR"][t, f, s] * (otmul if is_hol else 1)
    # overtime
    prod_cost += base * V["ProdO"][t, f, s]
    prod_cost += wage * hrs * V["ProdO"][t, f, s] * otmul

    co2_prod += (V["ProdR"][t, f, s] + V["ProdO"][t, f, s]) * dp.carbon_f[f]

# Transport cost & CO₂
for t, (f, h, mname) in V["ShipF2W"].keys():
    base = dp.COST_FC_WH[f, h, mname] + dp.BORDER_FC_WH[f, h]
    multi = 1.0
    if t in dp.BAD_WEATHER_DATES: multi *= 3
    if week_monday(t).to_period("W-MON") in dp.HIGH_OIL_WEEKS: multi *= 2
    qty = V["ShipF2W"][t, (f, h, mname)]      # containers
    trans_cost += base * multi * qty
    co2_tran   += dp.CO2_FC_WH[f, h, mname] * qty

for t, (w, c) in V["ShipW2C"].keys():
    base = dp.COST_WH_CT[w, c]
    multi = 1.0
    if t in dp.BAD_WEATHER_DATES: multi *= 3
    if week_monday(t).to_period("W-MON") in dp.HIGH_OIL_WEEKS: multi *= 2
    qty = V["ShipW2C"][t, (w, c)]
    trans_cost += base * multi * qty
    co2_tran   += dp.CO2_WH_CT[w, c] * qty

# Mode change penalty (5% of prior 4-week cost)
BLOCK_W = dp.oil_price["week"].unique().tolist()
for b,dow,(f,h) in V["modeBlock"].keys():
    if b == 0:
        continue
    prev_cost = gp.LinExpr()
    for t,(ff,hh,mn) in V["ShipF2W"].keys():
        if ff!=f or hh!=h or mn not in MODES_FCWH:
            continue
        widx = BLOCK_W.index(week_monday(t).to_period("W-MON"))
        if widx//dp.MODE_BLOCK_WEEKS == b-1 and (t.weekday() if isinstance(t, dt.date) else 0)==dow:
            base = dp.COST_FC_WH[f,h,mn] + dp.BORDER_FC_WH[f,h]
            mul = 1.0
            if t in dp.BAD_WEATHER_DATES: mul *= 3
            if week_monday(t).to_period("W-MON") in dp.HIGH_OIL_WEEKS: mul *= 2
            prev_cost += base * mul * V["ShipF2W"][t,(f,h,mn)]
    pen = mdl.addVar(lb=0.0, name=f"Pen_{b}_{dow}_{f}_{h}")
    mdl.addConstr(pen >= 0.05 * prev_cost - BIG_M*(1 - V["modeChange"][b,dow,(f,h)]))
    mdl.addConstr(pen <= 0.05 * prev_cost)
    mdl.addConstr(pen <= BIG_M * V["modeChange"][b,dow,(f,h)])
    change_pen += pen

# CO₂ Environmental fee
env_fee = TON_PENALTY_USD * ceil_div_expr(co2_prod + co2_tran, 1000)
mdl.setObjective(mdl.getObjective() + env_fee + change_pen)

# ═══════════════ OPTIMISE ═════════════════════════════════════════════════
print("⋆  Optimising (threads={}, weekly={}) …".format(
      args.threads, args.weekly))
mdl.optimize()

if mdl.SolCount == 0:
    raise RuntimeError("Model infeasible!")

print(f"✓ Optimised  –  Obj = {mdl.ObjVal:,.0f} USD, Gap = {mdl.MIPGap:.2%}")

# ═══════════════ DUMP TO SUBMISSION DB ════════════════════════════════════
print("⋆  Dumping plan_submission_template.db …")
TPL = f"{BASE_DIR}/plan_submission_template.db"
tmp = tempfile.NamedTemporaryFile(delete=False).name
shutil.copy(TPL, tmp)
con = sqlite3.connect(tmp); cur = con.cursor()
cur.execute("DELETE FROM plan_submission_template")

# Production rows (expand ProdR & ProdO)
for t, f, s in V["ProdR"].keys():
    date_str = t.isoformat() if isinstance(t, dt.date) else t.start_time.date().isoformat()
    cur.execute("""INSERT INTO plan_submission_template
        (date,factory,sku,production_qty,ot_qty)
        VALUES (?,?,?,?,?)""",
        (date_str, f, s,
         int(V["ProdR"][t, f, s].X),
         int(V["ProdO"][t, f, s].X)))

# Shipments F2W
for t, (f, h, mname) in V["ShipF2W"].keys():
    qty = int(V["ShipF2W"][t, (f, h, mname)].X) * CONTAINER_CAP
    if qty == 0: continue
    date_str = t.isoformat() if isinstance(t, dt.date) else t.start_time.date().isoformat()
    cur.execute("""INSERT INTO plan_submission_template
        (date,from_city,to_city,mode,ship_qty)
        VALUES (?,?,?,?,?)""",
        (date_str, f, h, mname, qty))

# Shipments W2C
for t, (w, c) in V["ShipW2C"].keys():
    qty = int(V["ShipW2C"][t, (w, c)].X) * CONTAINER_CAP
    if qty == 0: continue
    date_str = t.isoformat() if isinstance(t, dt.date) else t.start_time.date().isoformat()
    cur.execute("""INSERT INTO plan_submission_template
        (date,from_city,to_city,mode,ship_qty)
        VALUES (?,?,?,?,?)""",
        (date_str, w, c, "TRUCK", qty))

con.commit(); con.close()
print(f"🎉  Submission DB ready  →  {tmp}")
