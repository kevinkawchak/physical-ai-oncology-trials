# Section 4: Results

## 4.1 FAERS & Rounds 1-3

The FAERS dataset table above represents a sampling of the static FDA Adverse Event Reporting System (FAERS) dataset used in rounds. The reference solution table is the final reference solution that each competitor's function was seeking to output. After the \_1RD_Tournament_FAERS.ipynb was finalized, head-to-head code matchups were conducted for Rounds 1-3, as detailed in the Methods section. The winner of each round's Python notebook was used as the input along with Multi_Round_Prompt to enable iterative updates for subsequent rounds.

---

## 4.2 Round 4 Code Results

The 152 line Gpt-5.2-pro Round 4 detect_signal_emergence_improved function, which is based on its prior Round M notebook and Multi_Round_Prompt is provided in the following page to build drug–reaction pairs and compute PRR-like observed/expected per quarter. The function uses 9 algorithms for data cleaning/schema normalization, rule-based filtering, deterministic time-binning, relational join/entity resolution, frequency counting/contingency component estimation, observed–expected disproportionality, thresholding/decision rule, "first hitting time"/earliest-emergence selection, and post-processing/summarization.

**Multi_Round_Prompt LLM Thoughts Addressed**

- "Achieving a perfect score of 1.0 on all metrics is the goal"
- "To improve the quality from Code_A's score of 9.5 to 10 [prior Round 3 notebook's Gpt-5.2-pro code], I should add type hints, enhance docstrings, ensure vectorization, have a function count of at least three, and maintain a proper comment ratio"
- "I'm planning to create a function that includes type hints and input checking while avoiding external dependencies"

**Competition Code Implements Useful Comments**

1. "Matched the notebook Reference Solution exactly (per-quarter PRR via expected=(drug_cases*reac_cases)/total_cases, earliest-quarter emergence, PRR rounded to 3)."
2. "Hardened input handling (column normalization, strict FAERS event_dt→quarter parsing, safe early exits with correct empty schema)."
3. "Kept performance high and stable (column subsetting, vectorized groupby/nunique + merges; no Python loops over rows)."

---

### Gpt-5.2-pro Round 4 Code Submission, 152 Lines

```python
def detect_signal_emergence_improved(
    demo_df: pd.DataFrame, drug_df: pd.DataFrame, reac_df: pd.DataFrame, min_cases: int = 3, prr_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Improved implementation of Task 1 (TASK_1_SPEC).

    Improvements made:
    # 1. Matched the notebook Reference Solution exactly (per-quarter PRR via expected=(drug_cases*reac_cases)/total_cases, earliest-quarter emergence, PRR rounded to 3).
    # 2. Hardened input handling (column normalization, strict FAERS event_dt→quarter parsing, safe early exits with correct empty schema).
    # 3. Kept performance high and stable (column subsetting, vectorized groupby/nunique + merges; no Python loops over rows).

    Returns a DataFrame with columns:
    ['drug_name', 'reaction', 'emergence_quarter', 'emergence_prr', 'total_cases', 'is_signal']
    """
    required_out_cols = ["drug_name", "reaction", "emergence_quarter", "emergence_prr", "total_cases", "is_signal"]

    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=required_out_cols)

    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        out = df.copy(deep=False)
        out.columns = out.columns.astype(str).str.lower().str.strip()
        return out

    def _parse_quarter(demo: pd.DataFrame) -> pd.DataFrame:
        # Matches the notebook reference parsing behavior exactly.
        if demo.empty or ("primaryid" not in demo.columns) or ("event_dt" not in demo.columns):
            return pd.DataFrame(columns=["primaryid", "quarter"])

        d = demo[["primaryid", "event_dt"]].dropna(subset=["event_dt"]).copy()
        d["event_dt_str"] = d["event_dt"].astype(str).str.replace(".0", "", regex=False)
        d = d[d["event_dt_str"].str.len() >= 6]

        d["year"] = d["event_dt_str"].str[:4]
        d["month"] = pd.to_numeric(d["event_dt_str"].str[4:6], errors="coerce")
        d = d[d["month"].notna() & d["year"].astype(str).str.match(r"^\d{4}$", na=False)]
        d = d[(d["month"] >= 1) & (d["month"] <= 12)]

        d["quarter"] = d["year"].astype(str) + "Q" + (((d["month"] - 1) // 3 + 1).astype(int)).astype(str)
        return d[["primaryid", "quarter"]]

    # --- Normalize inputs (robust to column casing/whitespace) ---
    demo = _norm_cols(demo_df)
    drug = _norm_cols(drug_df)
    reac = _norm_cols(reac_df)
    if demo.empty or drug.empty or reac.empty:
        return _empty()

    # --- Suspect drug filtering + normalization (uppercase, trimmed) ---
    need_drug_cols = {"primaryid", "drugname", "role_cod"}
    need_reac_cols = {"primaryid", "pt"}
    if not need_drug_cols.issubset(drug.columns) or not need_reac_cols.issubset(reac.columns):
        return _empty()

    drugs = drug.loc[drug["role_cod"].isin(["PS", "SS"]), ["primaryid", "drugname"]].copy()
    if drugs.empty:
        return _empty()
    drugs["drugname"] = drugs["drugname"].astype(str).str.upper().str.strip()

    # --- Parse quarters from DEMO ---
    demo_q = _parse_quarter(demo)
    if demo_q.empty:
        return _empty()

    # --- Merge DRUG + REAC + DEMO on primaryid (inner joins as reference) ---
    merged = drugs.merge(reac[["primaryid", "pt"]], on="primaryid", how="inner").merge(
        demo_q, on="primaryid", how="inner"
    )
    if merged.empty:
        return _empty()

    # --- Per-quarter component counts (unique primaryid) ---
    quarter_totals = merged.groupby("quarter", sort=False).agg(total_cases=("primaryid", "nunique")).reset_index()
    drug_quarter = (
        merged.groupby(["drugname", "quarter"], sort=False)["primaryid"].nunique().reset_index(name="drug_cases")
    )
    reac_quarter = merged.groupby(["pt", "quarter"], sort=False)["primaryid"].nunique().reset_index(name="reac_cases")
    pair_quarter = (
        merged.groupby(["drugname", "pt", "quarter"], sort=False)["primaryid"].nunique().reset_index(name="pair_cases")
    )

    analysis = (
        pair_quarter.merge(drug_quarter, on=["drugname", "quarter"], how="left")
        .merge(reac_quarter, on=["pt", "quarter"], how="left")
        .merge(quarter_totals, on="quarter", how="left")
    )

    # Reference PRR definition used in notebook:
    # expected = (drug_cases * reac_cases) / total_cases ; prr = pair_cases / expected
    analysis["expected"] = (analysis["drug_cases"] * analysis["reac_cases"]) / analysis["total_cases"]
    analysis["prr"] = np.where(analysis["expected"] > 0, analysis["pair_cases"] / analysis["expected"], 0.0)

    # --- Signal emergence (first quarter meeting threshold) ---
    signals = analysis[(analysis["prr"] >= float(prr_threshold)) & (analysis["pair_cases"] >= int(min_cases))].copy()
    if signals.empty:
        return _empty()

    signals = signals.sort_values("quarter", ascending=True)
    first_emergence = signals.groupby(["drugname", "pt"]).first().reset_index()

    total_by_pair = (
        merged.groupby(["drugname", "pt"], sort=False)["primaryid"].nunique().reset_index(name="total_cases_all")
    )

    result_df = first_emergence[["drugname", "pt", "quarter", "prr"]].merge(
        total_by_pair, on=["drugname", "pt"], how="left"
    )

    result_df = result_df.rename(
        columns={
            "drugname": "drug_name",
            "pt": "reaction",
            "quarter": "emergence_quarter",
            "prr": "emergence_prr",
            "total_cases_all": "total_cases",
        }
    )
    result_df["emergence_prr"] = result_df["emergence_prr"].astype(float).round(3)
    result_df["total_cases"] = result_df["total_cases"].fillna(0).astype(int)
    result_df["is_signal"] = True

    return result_df[required_out_cols].sort_values("emergence_prr", ascending=False).reset_index(drop=True)


# REQUIRED: Set result variable
result = detect_signal_emergence_improved(demo_df, drug_df, reac_df)
print(f"Found {len(result)} signals")
```
