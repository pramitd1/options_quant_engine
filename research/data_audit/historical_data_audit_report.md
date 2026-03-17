# NIFTY Historical Data Audit Report
**Generated:** 2026-03-17T18:17:43.940426
**Overall Data Quality Score: 88.7 / 100**

## 1. Data Inventory & Structure

### 1.1 Raw Data Files
- **Raw CSV count:** 3501
- **Date range:** 2012-01-02 → 2026-03-16

### 1.2 Processed Parquet Files (NIFTY)

| Year | Rows | Trading Days | Columns | CE Rows | PE Rows | FUT Rows | Has LTP | Has Underlying |
|------|------|-------------|---------|---------|---------|----------|---------|----------------|
| 2012 | 356,837 | 247 | 14 | 178,048 | 178,048 | 741 | No | No |
| 2013 | 385,818 | 246 | 14 | 192,540 | 192,540 | 738 | No | No |
| 2014 | 475,697 | 243 | 14 | 237,484 | 237,484 | 729 | No | No |
| 2015 | 533,777 | 247 | 14 | 266,518 | 266,518 | 741 | No | No |
| 2016 | 514,988 | 246 | 14 | 257,125 | 257,125 | 738 | No | No |
| 2017 | 528,158 | 248 | 14 | 263,707 | 263,707 | 744 | No | No |
| 2018 | 567,188 | 246 | 14 | 283,225 | 283,225 | 738 | No | No |
| 2019 | 770,822 | 244 | 14 | 385,045 | 385,045 | 732 | No | No |
| 2020 | 732,389 | 250 | 14 | 360,044 | 371,595 | 750 | No | No |
| 2021 | 511,306 | 247 | 14 | 245,707 | 264,858 | 741 | No | No |
| 2022 | 562,785 | 248 | 14 | 276,798 | 285,243 | 744 | No | No |
| 2023 | 339,804 | 245 | 14 | 169,304 | 169,765 | 735 | No | No |
| 2024 | 394,888 | 246 | 16 | 195,411 | 198,739 | 738 | Yes | Yes |
| 2025 | 395,027 | 248 | 16 | 195,342 | 198,941 | 744 | Yes | Yes |
| 2026 | 82,359 | 50 | 16 | 40,456 | 41,753 | 150 | Yes | Yes |
| **TOTAL** | **7,151,843** | **3501** | — | — | — | — | — | — |

### 1.3 Spot Data
- **Rows:** 4,537
- **Date range:** 2007-09-17 → 2026-03-17
- **Columns:** ['date', 'open', 'high', 'low', 'close', 'volume']
- **Zero volume days:** 1334

### 1.4 Schema Changes Detected

- 2012→2013: Column 'change_in_oi' type changed from float64 to int64
- 2012→2013: Column 'open_interest' type changed from float64 to int64
- 2013→2014: Column 'contracts' type changed from float64 to int64
- 2018→2019: Column 'change_in_oi' type changed from int64 to float64
- 2018→2019: Column 'contracts' type changed from int64 to float64
- 2018→2019: Column 'open_interest' type changed from int64 to float64
- 2019→2020: Column 'change_in_oi' type changed from float64 to int64
- 2019→2020: Column 'contracts' type changed from float64 to int64
- 2019→2020: Column 'open_interest' type changed from float64 to int64
- 2020→2021: Column 'change_in_oi' type changed from int64 to float64
- 2020→2021: Column 'contracts' type changed from int64 to float64
- 2020→2021: Column 'open_interest' type changed from int64 to float64
- 2021→2022: Column 'change_in_oi' type changed from float64 to int64
- 2021→2022: Column 'contracts' type changed from float64 to int64
- 2021→2022: Column 'open_interest' type changed from float64 to int64
- 2022→2023: Column 'change_in_oi' type changed from int64 to float64
- 2022→2023: Column 'contracts' type changed from int64 to float64
- 2022→2023: Column 'open_interest' type changed from int64 to float64
- 2023→2024: Added columns ['last_price', 'underlying_price']
- 2023→2024: Column 'change_in_oi' type changed from float64 to int64
- 2023→2024: Column 'contracts' type changed from float64 to int64
- 2023→2024: Column 'open_interest' type changed from float64 to int64
- 2023→2024: Instrument values changed from ['FUTIDX', 'OPTIDX'] to ['FUTIDX', 'IDF', 'IDO', 'OPTIDX']
- 2024→2025: Instrument values changed from ['FUTIDX', 'IDF', 'IDO', 'OPTIDX'] to ['IDF', 'IDO']
- Raw CSV format change detected at 2025-01-01: 34 columns vs previous 16 columns

## 2. Data Quality Audit

### 2.1 Missing Trading Days
- **Missing days (spot exists but no FO data):** 3
  - 2013-07-08
  - 2013-10-09
  - 2021-03-30

### 2.2 Data Anomalies

- **Total anomalies:** 316
- **By severity:** CRITICAL=13, WARNING=273, INFO=30

| Anomaly Type | Count | Severity |
|---|---|---|
| DUPLICATE_ROWS | 273 | WARNING |
| EXTREME_OI_CHANGE | 15 | INFO |
| CLOSE_SETTLE_MISMATCH | 15 | INFO |
| NAT_EXPIRY_DATE | 13 | CRITICAL |

### 2.3 Critical Issues

- **NAT_EXPIRY_DATE**: {"year": 2012, "count": 32182, "pct": "9.0%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2013, "count": 34216, "pct": "8.9%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2014, "count": 41568, "pct": "8.8%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2015, "count": 38696, "pct": "7.3%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2016, "count": 38968, "pct": "7.6%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2017, "count": 36914, "pct": "7.0%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2018, "count": 51156, "pct": "9.0%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2019, "count": 65482, "pct": "8.5%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2020, "count": 61352, "pct": "8.4%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2021, "count": 30118, "pct": "5.9%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2022, "count": 32607, "pct": "5.8%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2023, "count": 18485, "pct": "5.5%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}
- **NAT_EXPIRY_DATE**: {"year": 2024, "count": 21938, "pct": "5.6%", "note": "Rows with missing expiry_date \u2014 likely from preprocessing parse failures. These rows have valid strikes and OHLC but no expiry, making them unusable for IV computation and chain construction without recovery."}

## 3. Options Chain Integrity

- **Total integrity issues:** 513

| Issue Type | Count |
|---|---|
| UNPAIRED_STRIKES | 513 |

## 4. Point-in-Time Consistency

- **Total PIT issues:** 15

| Issue Type | Count | Severity |
|---|---|---|
| EXTREME_SETTLE_CLOSE_RATIO | 15 | WARNING |

### PIT Assessment

- **Data source:** NSE Bhav Copy (official End-of-Day settlement data)
- **Granularity:** Daily (no intraday data in raw files)
- **Lookahead risk:** LOW — data represents known EOD values
- **IV status:** Not in raw data — must be reconstructed from available fields
- **Greeks status:** Not in raw data — must be computed from IV
- **Recommendation:** All derived features (IV, Greeks, signals) must be computed using only same-day or prior-day inputs

## 5. Feature Reconstruction Plan

| Feature | Status | Priority | Reconstruction Method |
|---|---|---|---|
| Implied Volatility (IV) | MISSING_IN_RAW | CRITICAL | Newton-Raphson on Black-Scholes using: close, strike_price, underlying_price/spo... |
| Greeks (Delta, Gamma, Theta, Vega, Rho) | MISSING_IN_RAW | HIGH | Black-Scholes closed-form from IV, spot, strike, T, r... |
| Spot Price (underlying_price) | PARTIALLY_AVAILABLE | CRITICAL | Available in 2025-2026 parquets. For 2012-2024: join with spot_daily parquet or ... |
| Last Price (last_price) | PARTIALLY_AVAILABLE | LOW | Available in 2024-2026 parquets. For 2012-2023: use close as proxy (bhav copy cl... |
| Days to Expiry | DERIVABLE | HIGH | (expiry_date - trade_date).days, floor at 0... |
| Moneyness | DERIVABLE | MEDIUM | strike_price / spot for CE, spot / strike_price for PE... |
| Volume (totalTradedVolume) | AVAILABLE_AS_CONTRACTS | MEDIUM | contracts column = number of contracts traded. Multiply by lot size for notional... |

## 6. Data Normalization Requirements

### Canonical Schema (target)

```
  trade_date
  symbol
  instrument
  expiry_date
  strike_price
  option_type
  open
  high
  low
  close
  last_price
  settle_price
  underlying_price
  contracts
  open_interest
  change_in_oi
```

### Required Transformations

1. **Column unification:** 14→16 columns for 2012-2023 (add `last_price`, `underlying_price`)
2. **Type normalization:** `contracts` float64→int64 for 2012; `trade_date` to datetime
3. **Spot price fill:** Join `underlying_price` from spot_daily for pre-2025 data
4. **Instrument codes:** Unify `OPTIDX`/`IDO` → standard label
5. **Expiry format:** Ensure all `expiry_date` as datetime with consistent timezone handling
6. **option_type:** Standardize `XX`→`FUT`

## 7. Replay-Ready Data Pipeline

### Architecture

```
replay_historical_snapshot(date: str, symbol: str = 'NIFTY')
  → {
      'spot': {'date': ..., 'open': ..., 'high': ..., 'low': ..., 'close': ...},
      'option_chain': pd.DataFrame with canonical columns + IV + Greeks,
      'quality_score': float,
      'metadata': {'data_source': 'NSE_BHAV', 'granularity': 'EOD', ...}
    }
```

### Point-in-Time Guarantees

1. Only uses data for the requested date (no future data)
2. IV computed from same-day close prices
3. Greeks computed from same-day IV
4. Spot from same-day spot_daily close
5. Deterministic output (same input → same output)

## 8. Data Quality Scores

| Year | Avg Score | Good Days (≥80) | Partial (50-79) | Poor (<50) | Min | Max |
|------|----------|-----------------|-----------------|------------|-----|-----|
| 2012 | 85.6 | 224 (90.7%) | 23 (9.3%) | 0 (0.0%) | 72.0 | 87.0 |
| 2013 | 85.4 | 222 (90.2%) | 24 (9.8%) | 0 (0.0%) | 67.0 | 87.0 |
| 2014 | 84.6 | 219 (90.1%) | 24 (9.9%) | 0 (0.0%) | 72.0 | 87.0 |
| 2015 | 82.9 | 228 (92.3%) | 19 (7.7%) | 0 (0.0%) | 67.0 | 87.0 |
| 2016 | 84.0 | 226 (91.9%) | 20 (8.1%) | 0 (0.0%) | 67.0 | 87.0 |
| 2017 | 84.1 | 229 (92.3%) | 19 (7.7%) | 0 (0.0%) | 67.0 | 87.0 |
| 2018 | 84.3 | 222 (90.2%) | 24 (9.8%) | 0 (0.0%) | 67.0 | 87.0 |
| 2019 | 83.9 | 217 (88.9%) | 27 (11.1%) | 0 (0.0%) | 60.0 | 87.0 |
| 2020 | 84.6 | 232 (92.8%) | 18 (7.2%) | 0 (0.0%) | 60.0 | 92.0 |
| 2021 | 88.8 | 228 (92.3%) | 19 (7.7%) | 0 (0.0%) | 72.0 | 100.0 |
| 2022 | 94.4 | 245 (98.8%) | 3 (1.2%) | 0 (0.0%) | 77.0 | 100.0 |
| 2023 | 98.8 | 245 (100.0%) | 0 (0.0%) | 0 (0.0%) | 85.0 | 100.0 |
| 2024 | 98.7 | 246 (100.0%) | 0 (0.0%) | 0 (0.0%) | 85.0 | 100.0 |
| 2025 | 100.0 | 248 (100.0%) | 0 (0.0%) | 0 (0.0%) | 100.0 | 100.0 |
| 2026 | 100.0 | 50 (100.0%) | 0 (0.0%) | 0 (0.0%) | 100.0 | 100.0 |

## 9. Data Usability Summary

- **Total trading days audited:** 3501
- **Usable (score ≥ 80):** 3281 (93.7%)
- **Requires cleaning (50-79):** 220 (6.3%)
- **Unusable (< 50):** 0 (0.0%)

## 10. Recommendations

### Critical Actions

1. **Build normalization pipeline** — Unify all 15 year-files into canonical 16-column schema
2. **Fill underlying_price** — Join spot_daily close for 2012-2024 data
3. **Compute IV** — Use Newton-Raphson BS solver (already in `historical_data_adapter.py`)
4. **Build replay loader module** — `replay_historical_snapshot(date)` function
5. **Handle zero-close rows** — Use settle_price as fallback; flag as low-confidence

### Quality Improvements

6. Deduplicate rows per (date, strike, option_type, expiry)
7. Flag and quarantine days with quality score < 50
8. Validate strike grid continuity for ATM±10 strikes
9. Track NIFTY lot size changes for accurate volume normalization

### Data Gaps

- 2012→2013: Column 'change_in_oi' type changed from float64 to int64
- 2012→2013: Column 'open_interest' type changed from float64 to int64
- 2013→2014: Column 'contracts' type changed from float64 to int64
- 2018→2019: Column 'change_in_oi' type changed from int64 to float64
- 2018→2019: Column 'contracts' type changed from int64 to float64
