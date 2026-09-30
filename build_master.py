#!/usr/bin/env python3
"""Build NJ Housing Affordability master dataset.
Merges: Zillow ZHVI (municipality-aggregated), NZLUD zoning, NJGIN boundaries
attributes (official names/codes), ACS 2023 median household income.
"""
import json, re, glob
import pandas as pd
import numpy as np

UP = "/sessions/youthful-clever-fermat/mnt/uploads"
OUT = "/sessions/youthful-clever-fermat/mnt/outputs"
TR = glob.glob("/sessions/youthful-clever-fermat/mnt/.claude/projects/*/*/tool-results")[0]

# ---------- 1. ZHVI: resolve ambiguities ----------
z = pd.read_csv(f"{UP}/zhvi_NJ_with_codes-2 - zhvi_NJ_with_codes-2.csv")
# RegionID-based resolutions (boro vs twp):
#  3752  "Boonton"    -> 1401 Boonton Town ("Boonton Township" RegionID 17117 already holds 1402)
#  831875 "Andover"   -> 1902 Andover Township (NZLUD GEOID 3401360 = township; borough too small)
#  760887 "Berlin"    -> keep 405 Berlin Borough ("West Berlin" RegionID 21762 holds twp 406)
#  10869  "Clinton"   -> keep 1005 Clinton Town
#  397046 "Shrewsbury"-> keep 1344 Shrewsbury Borough (matches NZLUD)
z.loc[z.RegionID == 3752, "municipality_code"] = 1401
z.loc[z.RegionID == 831875, "municipality_code"] = 1902
resolution_notes = {
    760887: "Ambiguous Berlin -> Borough (405); West Berlin holds Twp 406",
    3752: "Ambiguous Boonton -> Town (1401); Boonton Township holds 1402",
    10869: "Ambiguous Clinton -> Town (1005)",
    397046: "Ambiguous Shrewsbury -> Borough (1344); NZLUD match",
    831875: "Ambiguous Andover -> Township (1902); NZLUD match",
}

month_cols = [c for c in z.columns if re.match(r"^\d{4}-\d{2}-\d{2}$", c)]
LATEST = month_cols[-1]  # 2026-06-30

# split regions contribute to BOTH municipalities
rows = []
for _, r in z.iterrows():
    rows.append((int(r.municipality_code), r))
    if pd.notna(r.SecondaryMunicipality_code):
        rows.append((int(r.SecondaryMunicipality_code), r))

groups = {}
for code, r in rows:
    groups.setdefault(code, []).append(r)

def june(year):
    c = f"{year}-06-30"
    return c if c in month_cols else None

recs = []
for code, members in groups.items():
    dfm = pd.DataFrame(members)
    series = dfm[month_cols].mean(axis=0, skipna=True)  # equal-weight avg across Zillow regions
    rec = {"municipality_code": code,
           "n_zillow_regions": len(members),
           "zillow_regions": "; ".join(dfm.RegionName.tolist()),
           "zillow_region_ids": "; ".join(str(i) for i in dfm.RegionID.tolist()),
           "resolution_note": "; ".join(resolution_notes[i] for i in dfm.RegionID if i in resolution_notes) or "",
           "zhvi_latest": series[LATEST]}
    for y in range(2000, 2027):
        c = june(y)
        rec[f"zhvi_june_{y}"] = series[c] if c else np.nan
    recs.append(rec)
zm = pd.DataFrame(recs)
zm["zhvi_growth_1yr"] = zm.zhvi_latest / zm.zhvi_june_2025 - 1
zm["zhvi_growth_5yr"] = zm.zhvi_latest / zm.zhvi_june_2021 - 1
zm["zhvi_growth_10yr"] = zm.zhvi_latest / zm.zhvi_june_2016 - 1
zm["zhvi_cagr_2000"] = (zm.zhvi_latest / zm.zhvi_june_2000) ** (1/26.0) - 1

# ---------- 2. Boundaries attribute table ----------
raw = open(f"{TR}/mcp-workspace-web_fetch-1784403982848.txt").read()
j = json.loads(raw[raw.index('{"objectIdFieldName"'):]) if '{"objectIdFieldName"' in raw else json.loads(raw[raw.index("{"):])
b = pd.DataFrame([f["attributes"] for f in j["features"]])
b["municipality_code"] = b.MUN_CODE.astype(int)
b = b.rename(columns={"NAME": "municipality", "MUN_TYPE": "mun_type", "COUNTY": "county",
                      "CENSUS2020": "cousub_geoid", "POP2020": "pop_2020"})
b = b[["municipality_code", "municipality", "mun_type", "county", "cousub_geoid", "pop_2020"]]
assert b.municipality_code.is_unique

# ---------- 3. ACS income ----------
tj = json.load(open(f"{TR}/toolu_01L6FK2VjUMZSndvbYAwcxHH.json"))
txt = tj[0]["text"]
acs = json.loads(txt[txt.index('{"response"'):])["response"]["data"]
hdr = acs[0]
ai = pd.DataFrame(acs[1:], columns=hdr)
ai["cousub_geoid"] = ai.GEO_ID.str[-10:]
ai["median_hh_income"] = pd.to_numeric(ai.B19013_001E, errors="coerce")
ai.loc[ai.median_hh_income < 0, "median_hh_income"] = np.nan
ai = ai[["cousub_geoid", "median_hh_income"]]

# ---------- 4. NZLUD ----------
n = pd.read_csv(f"{UP}/nzlud_muni_NJ_with_codes - nzlud_muni_NJ_with_codes.csv")
n = n.replace("NA", np.nan)
nz_keep = ["municipality_code", "GEOID", "place", "type", "restrict_sf_permit", "restrict_mf_permit",
           "min_lot_size", "half_acre_less", "half_acre_more", "one_acre_more", "two_acre_more",
           "open_space", "inclusionary", "adu", "height_ft_median", "height_st_median",
           "parking_median", "mf_per", "total_nz", "total_rz", "zri", "zri_st"]
nn = n[nz_keep].rename(columns={"GEOID": "nzlud_geoid", "place": "nzlud_place", "type": "nzlud_type"})
nn["zri"] = pd.to_numeric(nn.zri, errors="coerce")
nn["zri_st"] = pd.to_numeric(nn.zri_st, errors="coerce")
assert nn.municipality_code.is_unique

# ---------- 4b. Redfin city-level market metrics (NJ-only file) ----------
rf_raw = pd.read_csv(f"{OUT}/redfin_housing_market_monthly_NJ_cities_2023_Jan_to_2026_Jun.csv")
rf_raw = rf_raw[rf_raw["PERIOD BEGIN"] == rf_raw["PERIOD BEGIN"].max()].copy()  # latest rolling-3mo window
rf_raw["city"] = rf_raw["REGION NAME"].str.replace(", NJ", "", regex=False)
for c in ["MEDIAN SALE PRICE NSA ($)", "MEDIAN DAYS ON MARKET (DAYS)", "AVERAGE SALE TO LIST RATIO (%)",
          "SHARE SOLD ABOVE ORIGINAL LIST (%)", "INVENTORY", "HOMES SOLD"]:
    rf_raw[c] = pd.to_numeric(rf_raw[c], errors="coerce")

# name -> muni code: (a) via user's curated Zillow crosswalk, (b) unique official base name
zx = z.drop_duplicates("RegionName").set_index("RegionName")["municipality_code"].to_dict()
base = b.copy()
base["basename"] = base.municipality.str.replace(
    r"\s+(Township|Borough|City|Town|Village)$", "", regex=True).str.strip()
counts = base.basename.value_counts()
uniq = base[base.basename.isin(counts[counts == 1].index)].set_index("basename")["municipality_code"].to_dict()

def match_city(name):
    if name in zx: return int(zx[name]), "zillow-crosswalk"
    if name in uniq: return int(uniq[name]), "unique-name"
    return None, None
rf_raw[["municipality_code", "match_via"]] = rf_raw.city.apply(lambda n: pd.Series(match_city(n)))
matched = rf_raw.dropna(subset=["municipality_code"]).copy()
matched["municipality_code"] = matched.municipality_code.astype(int)
print(f"Redfin: {len(rf_raw)} NJ regions -> {matched.municipality_code.nunique()} municipalities matched "
      f"({(rf_raw.municipality_code.isna()).sum()} unmatched localities/CDPs skipped)")

def agg_rf(g):
    w = g["HOMES SOLD"].fillna(0)
    def wavg(col):
        v = g[col]; ok = v.notna() & (w > 0)
        return (v[ok] * w[ok]).sum() / w[ok].sum() if ok.any() else (v.mean() if v.notna().any() else np.nan)
    return pd.Series({
        "rf_median_sale_price": wavg("MEDIAN SALE PRICE NSA ($)"),
        "rf_days_on_market": wavg("MEDIAN DAYS ON MARKET (DAYS)"),
        "rf_sale_to_list": wavg("AVERAGE SALE TO LIST RATIO (%)"),
        "rf_share_above_list": wavg("SHARE SOLD ABOVE ORIGINAL LIST (%)"),
        "rf_inventory": g["INVENTORY"].sum(min_count=1),
        "rf_homes_sold": g["HOMES SOLD"].sum(min_count=1),
    })
rf = matched.groupby("municipality_code").apply(agg_rf).reset_index()

# ---------- 5. Merge (base = all 564 official municipalities) ----------
m = b.merge(ai, on="cousub_geoid", how="left") \
     .merge(zm, on="municipality_code", how="left") \
     .merge(nn, on="municipality_code", how="left") \
     .merge(rf, on="municipality_code", how="left")

# orphan codes present in data but not in official list?
orphans = (set(zm.municipality_code) | set(nn.municipality_code)) - set(b.municipality_code)
print("orphan codes (in data, not in boundaries):", sorted(orphans))

m["price_to_income"] = m.zhvi_latest / m.median_hh_income

# ---------- 6. Affordability index ----------
# Components as affordability percentiles (0-100, higher = MORE affordable):
#   P: price-to-income ratio (inverted)  weight .50
#   G: 5-yr ZHVI growth (inverted)       weight .25
#   Z: NZLUD zri zoning restrictiveness (inverted) weight .25
def aff_pct(s, invert=True):
    r = s.rank(pct=True) * 100
    return 100 - r if invert else r

m["pct_price_income"] = aff_pct(m.price_to_income)
m["pct_growth_5yr"] = aff_pct(m.zhvi_growth_5yr)
m["pct_zoning"] = aff_pct(m.zri)

W = {"pct_price_income": .50, "pct_growth_5yr": .25, "pct_zoning": .25}
def composite(row):
    avail = {k: w for k, w in W.items() if pd.notna(row[k])}
    if not avail or "pct_price_income" not in avail:  # require the core component
        return np.nan
    tw = sum(avail.values())
    return sum(row[k] * w for k, w in avail.items()) / tw
m["affordability_index"] = m.apply(composite, axis=1).round(2)
m["index_rank"] = m.affordability_index.rank(ascending=False, method="min")
m["data_coverage"] = (m.zhvi_latest.notna().astype(int).astype(str) + m.median_hh_income.notna().astype(int).astype(str) + m.zri.notna().astype(int).astype(str)).map(
    lambda s: {"111": "full", "110": "zhvi+income", "101": "zhvi+zoning", "011": "income+zoning", "100": "zhvi only", "010": "income only", "001": "zoning only", "000": "none"}[s])

m = m.sort_values("affordability_index", ascending=False)
num_cols = m.select_dtypes(include=[np.number]).columns
m[num_cols] = m[num_cols].round(4)
m.to_csv(f"{OUT}/nj_affordability_master.csv", index=False)

# ---------- 7. Slim JSON for the web app ----------
app = {}
for _, r in m.iterrows():
    code = f"{int(r.municipality_code):04d}"
    app[code] = {
        "name": r.municipality, "type": r.mun_type, "county": r.county,
        "pop": None if pd.isna(r.pop_2020) else int(r.pop_2020),
        "income": None if pd.isna(r.median_hh_income) else int(r.median_hh_income),
        "zhvi": None if pd.isna(r.zhvi_latest) else round(float(r.zhvi_latest)),
        "pti": None if pd.isna(r.price_to_income) else round(float(r.price_to_income), 2),
        "g1": None if pd.isna(r.zhvi_growth_1yr) else round(float(r.zhvi_growth_1yr) * 100, 1),
        "g5": None if pd.isna(r.zhvi_growth_5yr) else round(float(r.zhvi_growth_5yr) * 100, 1),
        "g10": None if pd.isna(r.zhvi_growth_10yr) else round(float(r.zhvi_growth_10yr) * 100, 1),
        "zri": None if pd.isna(r.zri) else round(float(r.zri), 2),
        "minlot": None if pd.isna(r.min_lot_size) else int(r.min_lot_size),
        "acre1": None if pd.isna(r.one_acre_more) else int(r.one_acre_more),
        "incl": None if pd.isna(r.inclusionary) else int(r.inclusionary),
        "adu": None if pd.isna(r.adu) else int(r.adu),
        "park": None if pd.isna(r.parking_median) else float(r.parking_median),
        "mfper": None if pd.isna(r.mf_per) else round(float(r.mf_per) * 100, 1),
        "pp": None if pd.isna(r.pct_price_income) else round(float(r.pct_price_income), 1),
        "pg": None if pd.isna(r.pct_growth_5yr) else round(float(r.pct_growth_5yr), 1),
        "pz": None if pd.isna(r.pct_zoning) else round(float(r.pct_zoning), 1),
        "idx": None if pd.isna(r.affordability_index) else float(r.affordability_index),
        "rank": None if pd.isna(r.index_rank) else int(r.index_rank),
        "cov": r.data_coverage,
        "rfp": None if pd.isna(r.rf_median_sale_price) else round(float(r.rf_median_sale_price)),
        "rfd": None if pd.isna(r.rf_days_on_market) else round(float(r.rf_days_on_market)),
        "rfs": None if pd.isna(r.rf_sale_to_list) else round(float(r.rf_sale_to_list), 1),
        "rfa": None if pd.isna(r.rf_share_above_list) else round(float(r.rf_share_above_list), 1),
        "rfi": None if pd.isna(r.rf_inventory) else int(r.rf_inventory),
        "rfn": None if pd.isna(r.rf_homes_sold) else int(r.rf_homes_sold),
        "regions": "" if pd.isna(r.zillow_regions) else r.zillow_regions,
        "ts": [None if pd.isna(r[f"zhvi_june_{y}"]) else round(float(r[f"zhvi_june_{y}"])) for y in range(2000, 2027)],
    }
json.dump(app, open(f"{OUT}/app_data.json", "w"))

print("master rows:", len(m))
print("with index:", m.affordability_index.notna().sum())
print("coverage:", m.data_coverage.value_counts().to_dict())
print("\nTOP 10 MOST AFFORDABLE:")
print(m.head(10)[["municipality", "county", "zhvi_latest", "median_hh_income", "price_to_income", "zri", "affordability_index"]].to_string(index=False))
print("\nBOTTOM 5:")
print(m.dropna(subset=["affordability_index"]).tail(5)[["municipality", "county", "zhvi_latest", "median_hh_income", "price_to_income", "affordability_index"]].to_string(index=False))
