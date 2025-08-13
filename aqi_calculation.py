"""
EPA AQI Calculation Module (US EPA Method)
==========================================

- Correct EPA linear interpolation
- Safe breakpoint selection (sorted)
- Proper unit handling (configurable per column)
- O3 8-hour and optional 1-hour handling (uses max per EPA guidance)
- Clamp at 500 (default) or allow >500 for research
- Row-wise DataFrame helper + expanded validation

Units expected by breakpoint tables:
- PM2.5, PM10: μg/m³
- O3: ppm (8-hour table; optional 1-hour table for high values)
- NO2: ppb (1-hour)
- CO: ppm (8-hour)
- SO2: ppb (1-hour)

Author: Data Science Team
Date: 2025-08-13
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd


# -------------------
# Breakpoint tables
# -------------------
# Tuples: (BP_lo, BP_hi, I_lo, I_hi)

BP_PM25 = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]

BP_PM10 = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

# O3: EPA guidance uses 8-hr table up to 0.200 ppm.
BP_O3_8H = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]

# O3 1-hour table (used for very high O3; optional, if you have 1-hr values)
BP_O3_1H = [
    (0.125, 0.164, 101, 150),
    (0.165, 0.204, 151, 200),
    (0.205, 0.404, 201, 300),
    (0.405, 0.504, 301, 400),
    (0.505, 0.604, 401, 500),
]

# NO2 (1-hr) expects ppb
BP_NO2_1H_PPB = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

# CO (8-hr) expects ppm
BP_CO_8H = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

# SO2 (1-hr) expects ppb
BP_SO2_1H_PPB = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]


AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def _interp_aqi(C: float, bps: List[Tuple[float, float, int, int]]) -> float:
    """Piecewise-linear interpolation on EPA breakpoints. Returns NaN if out of range."""
    if C is None or np.isnan(C) or C < 0:
        return np.nan
    for BP_lo, BP_hi, I_lo, I_hi in bps:
        if BP_lo <= C <= BP_hi:
            return ((I_hi - I_lo) / (BP_hi - BP_lo)) * (C - BP_lo) + I_lo
    return np.nan


def aqi_category(aqi: float) -> Optional[str]:
    if aqi is None or np.isnan(aqi):
        return None
    aqi_r = round(aqi)
    for lo, hi, name in AQI_CATEGORIES:
        if lo <= aqi_r <= hi:
            return name
    return "Hazardous+"  # if you allow >500


class EPAAQICalculator:
    """
    EPA-standard AQI calculation with unit handling and optional O3 1-hour support.

    Parameters
    ----------
    clamp_to_500 : bool
        If True (default), final AQI sub-indices are capped at 500 (EPA reporting style).
        If False, allows >500 for research/analytics.
    """

    def __init__(self, clamp_to_500: bool = True):
        self.clamp_to_500 = clamp_to_500

    # ------------------
    # Sub-index methods
    # ------------------

    def si_pm25(self, pm25_ugm3: float) -> float:
        return self._finalize(_interp_aqi(pm25_ugm3, BP_PM25))

    def si_pm10(self, pm10_ugm3: float) -> float:
        return self._finalize(_interp_aqi(pm10_ugm3, BP_PM10))

    def si_o3(self, o3_8h_ppm: Optional[float] = None, o3_1h_ppm: Optional[float] = None) -> float:
        """
        Compute O3 sub-index. If both 8-hr and 1-hr values provided, returns the max applicable.
        EPA practice: use 8-hr for 0–300 AQI range; 1-hr table used for higher O3 when available.
        """
        candidates: List[float] = []
        if o3_8h_ppm is not None and not np.isnan(o3_8h_ppm):
            candidates.append(_interp_aqi(o3_8h_ppm, BP_O3_8H))
        if o3_1h_ppm is not None and not np.isnan(o3_1h_ppm):
            candidates.append(_interp_aqi(o3_1h_ppm, BP_O3_1H))
        if not candidates:
            return np.nan
        return self._finalize(np.nanmax(candidates))

    def si_no2(self, no2_val: float, unit: str = "ppm") -> float:
        """
        NO2 expects ppb breakpoints. Convert if needed.
        unit: "ppm" or "ppb"
        """
        if no2_val is None or np.isnan(no2_val) or no2_val < 0:
            return np.nan
        no2_ppb = no2_val * 1000.0 if unit.lower() == "ppm" else no2_val
        return self._finalize(_interp_aqi(no2_ppb, BP_NO2_1H_PPB))

    def si_co(self, co_ppm: float) -> float:
        return self._finalize(_interp_aqi(co_ppm, BP_CO_8H))

    def si_so2(self, so2_val: float, unit: str = "ppm") -> float:
        """
        SO2 expects ppb breakpoints. Convert if needed.
        unit: "ppm" or "ppb"
        """
        if so2_val is None or np.isnan(so2_val) or so2_val < 0:
            return np.nan
        so2_ppb = so2_val * 1000.0 if unit.lower() == "ppm" else so2_val
        return self._finalize(_interp_aqi(so2_ppb, BP_SO2_1H_PPB))

    def _finalize(self, aqi_val: float) -> float:
        """Apply rounding and optional clamping."""
        if aqi_val is None or np.isnan(aqi_val):
            return np.nan
        if self.clamp_to_500:
            return float(min(500, round(aqi_val)))
        return float(round(aqi_val))

    # ------------------
    # Overall AQI logic
    # ------------------

    def overall_aqi(
        self,
        *,
        pm2_5: Optional[float] = None,
        pm10: Optional[float] = None,
        o3_8h: Optional[float] = None,  # ppm
        o3_1h: Optional[float] = None,  # ppm
        no2: Optional[float] = None,
        no2_unit: str = "ppm",
        co: Optional[float] = None,     # ppm
        so2: Optional[float] = None,
        so2_unit: str = "ppm",
    ) -> float:
        """Overall AQI = max of valid sub-indices."""
        subs: List[float] = []

        if pm2_5 is not None:
            subs.append(self.si_pm25(pm2_5))
        if pm10 is not None:
            subs.append(self.si_pm10(pm10))
        if o3_8h is not None or o3_1h is not None:
            subs.append(self.si_o3(o3_8h_ppm=o3_8h, o3_1h_ppm=o3_1h))
        if no2 is not None:
            subs.append(self.si_no2(no2, unit=no2_unit))
        if co is not None:
            subs.append(self.si_co(co))
        if so2 is not None:
            subs.append(self.si_so2(so2, unit=so2_unit))

        subs = [s for s in subs if not np.isnan(s)]
        return float(np.nanmax(subs)) if subs else np.nan

    # ------------------
    # DataFrame helpers
    # ------------------

    def compute_aqi_for_dataframe(
        self,
        df: pd.DataFrame,
        *,
        colmap: Dict[str, str] = None,
        # Units for gases in your DataFrame columns:
        no2_unit: str = "ppm",
        so2_unit: str = "ppm",
        # O3 handling:
        o3_columns: Tuple[str, Optional[str]] = ("o3", None),  # (o3_8h_col, o3_1h_col or None)
    ) -> pd.DataFrame:
        """
        Compute sub-indices + overall AQI for each row.

        Parameters
        ----------
        df : DataFrame with pollutant columns
        colmap : mapping from canonical names to df columns.
                 Defaults assume df uses: 'pm2_5','pm10','o3','no2','co','so2' and optional 'o3_1h'
                 Example:
                     colmap = {
                        "pm2_5": "PM2.5",
                        "pm10": "PM10",
                        "o3": "O3_8h_ppm",
                        "o3_1h": "O3_1h_ppm",
                        "no2": "NO2_ppm",
                        "co": "CO_ppm",
                        "so2": "SO2_ppm",
                     }
        no2_unit : "ppm" or "ppb" for df[colmap["no2"]]
        so2_unit : "ppm" or "ppb" for df[colmap["so2"]]
        o3_columns : (o3_8h_col, o3_1h_col or None)

        Returns
        -------
        DataFrame with added columns:
          ['AQI_PM2_5','AQI_PM10','AQI_O3','AQI_NO2','AQI_CO','AQI_SO2','AQI_Overall','AQI_Category']
        """
        # Default column mapping
        if colmap is None:
            colmap = {
                "pm2_5": "pm2_5",
                "pm10": "pm10",
                "o3": "o3",
                "o3_1h": "o3_1h",
                "no2": "no2",
                "co": "co",
                "so2": "so2",
            }

        o3_8h_col, o3_1h_col = o3_columns
        if o3_8h_col is None:
            o3_8h_col = colmap.get("o3")

        # Prepare outputs
        out_cols = {
            "AQI_PM2_5": [],
            "AQI_PM10": [],
            "AQI_O3": [],
            "AQI_NO2": [],
            "AQI_CO": [],
            "AQI_SO2": [],
            "AQI_Overall": [],
            "AQI_Category": [],
        }

        # Row-wise (clear & dependable; vectorization possible if needed)
        for _, row in df.iterrows():
            v_pm25 = row[colmap["pm2_5"]] if colmap["pm2_5"] in df.columns else np.nan
            v_pm10 = row[colmap["pm10"]] if colmap["pm10"] in df.columns else np.nan

            v_o3_8h = row[o3_8h_col] if (o3_8h_col and o3_8h_col in df.columns) else np.nan
            v_o3_1h = row[o3_1h_col] if (o3_1h_col and o3_1h_col in df.columns) else np.nan

            v_no2 = row[colmap["no2"]] if colmap["no2"] in df.columns else np.nan
            v_co = row[colmap["co"]] if colmap["co"] in df.columns else np.nan
            v_so2 = row[colmap["so2"]] if colmap["so2"] in df.columns else np.nan

            si_pm25 = self.si_pm25(v_pm25) if pd.notna(v_pm25) else np.nan
            si_pm10 = self.si_pm10(v_pm10) if pd.notna(v_pm10) else np.nan
            si_o3 = self.si_o3(o3_8h_ppm=v_o3_8h, o3_1h_ppm=v_o3_1h)
            si_no2 = self.si_no2(v_no2, unit=no2_unit) if pd.notna(v_no2) else np.nan
            si_co = self.si_co(v_co) if pd.notna(v_co) else np.nan
            si_so2 = self.si_so2(v_so2, unit=so2_unit) if pd.notna(v_so2) else np.nan

            overall = float(np.nanmax([x for x in [si_pm25, si_pm10, si_o3, si_no2, si_co, si_so2] if not np.isnan(x)]) \
                            ) if any(not np.isnan(x) for x in [si_pm25, si_pm10, si_o3, si_no2, si_co, si_so2]) else np.nan

            out_cols["AQI_PM2_5"].append(si_pm25)
            out_cols["AQI_PM10"].append(si_pm10)
            out_cols["AQI_O3"].append(si_o3)
            out_cols["AQI_NO2"].append(si_no2)
            out_cols["AQI_CO"].append(si_co)
            out_cols["AQI_SO2"].append(si_so2)
            out_cols["AQI_Overall"].append(overall)
            out_cols["AQI_Category"].append(aqi_category(overall))

        return df.assign(**out_cols)

    # ------------------
    # Validation helpers
    # ------------------

    def validate_known_points(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Validate at category endpoints for all pollutants.
        Returns dict: pollutant -> list of (calculated, expected)
        """
        results: Dict[str, List[Tuple[float, float]]] = {}

        # PM2.5
        pm25_cases = [(12.0, 50), (35.4, 100), (55.4, 150), (150.4, 200), (250.4, 300), (500.4, 500)]
        results["pm2_5"] = [(self.si_pm25(c), exp) for c, exp in pm25_cases]

        # PM10
        pm10_cases = [(54, 50), (154, 100), (254, 150), (354, 200), (424, 300), (604, 500)]
        results["pm10"] = [(self.si_pm10(c), exp) for c, exp in pm10_cases]

        # O3 (8-hr)
        o3_8h_cases = [(0.054, 50), (0.070, 100), (0.085, 150), (0.105, 200), (0.200, 300)]
        results["o3_8h"] = [(self.si_o3(o3_8h_ppm=c), exp) for c, exp in o3_8h_cases]

        # O3 (1-hr)
        o3_1h_cases = [(0.164, 150), (0.204, 200), (0.404, 300), (0.504, 400), (0.604, 500)]
        results["o3_1h"] = [(self.si_o3(o3_1h_ppm=c), exp) for c, exp in o3_1h_cases]

        # NO2 (ppb)
        no2_cases_ppb = [(53, 50), (100, 100), (360, 150), (649, 200), (1249, 300), (1649, 400), (2049, 500)]
        results["no2"] = [(self.si_no2(c, unit="ppb"), exp) for c, exp in no2_cases_ppb]

        # CO (ppm)
        co_cases = [(4.4, 50), (9.4, 100), (12.4, 150), (15.4, 200), (30.4, 300), (40.4, 400), (50.4, 500)]
        results["co"] = [(self.si_co(c), exp) for c, exp in co_cases]

        # SO2 (ppb)
        so2_cases_ppb = [(35, 50), (75, 100), (185, 150), (304, 200), (604, 300), (804, 400), (1004, 500)]
        results["so2"] = [(self.si_so2(c, unit="ppb"), exp) for c, exp in so2_cases_ppb]

        return results

    # ------------------
    # Backwards-compatibility wrappers
    # ------------------
    
    def calculate_overall_aqi(self, pollutant_data: Dict[str, float]) -> float:
        """
        Backwards-compatible wrapper that applies the same unit conversions used
        by the legacy implementation and computes overall AQI.
        Expects pollutant_data keys: 'pm2_5','pm10','o3','no2','co','so2'.
        """
        # Extract with defaults
        pm2_5 = pollutant_data.get("pm2_5")
        pm10 = pollutant_data.get("pm10")
        o3 = pollutant_data.get("o3")
        no2 = pollutant_data.get("no2")
        co = pollutant_data.get("co")
        so2 = pollutant_data.get("so2")

        # Legacy unit conversions:
        # - O3: ppb -> ppm
        # - CO: divide by 100 (legacy dataset scaling)
        # - NO2, SO2: ppb -> ppm
        o3_ppm = (o3 / 1000.0) if (o3 is not None and not np.isnan(o3)) else None
        co_ppm = (co / 100.0) if (co is not None and not np.isnan(co)) else None
        no2_ppm = (no2 / 1000.0) if (no2 is not None and not np.isnan(no2)) else None
        so2_ppm = (so2 / 1000.0) if (so2 is not None and not np.isnan(so2)) else None

        return self.overall_aqi(
            pm2_5=pm2_5,
            pm10=pm10,
            o3_8h=o3_ppm,
            no2=no2_ppm, no2_unit="ppm",
            co=co_ppm,
            so2=so2_ppm, so2_unit="ppm",
        )
    
    def calculate_aqi_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Backwards-compatible wrapper that returns a Series of overall AQI values
        using the actual OWM units with correct conversions.

        Data assumptions (OWM API):
        - PM2.5, PM10: μg/m³ (use as-is)
        - CO, NO2, O3, SO2: μg/m³ → convert using 25°C, 1 atm
          ppb = μg/m³ × (24.45 / molecular_weight), ppm = ppb / 1000
        """
        aqi_values: List[float] = []

        # Molecular weights (g/mol)
        MW = {"co": 28.01, "no2": 46.01, "o3": 48.00, "so2": 64.07}

        for _, row in df.iterrows():
            pollutant_data: Dict[str, float] = {}

            # Extract pollutants with proper unit handling
            for pollutant in ["pm2_5", "pm10", "o3", "no2", "co", "so2"]:
                if pollutant in df.columns:
                    value = row[pollutant]

                    # Skip invalid values
                    if pd.isna(value) or value < 0:
                        continue

                    # Apply unit conversions
                    if pollutant in ("pm2_5", "pm10"):
                        pollutant_data[pollutant] = float(value)
                    elif pollutant in ("o3", "no2", "co", "so2"):
                        try:
                            ppb = float(value) * (24.45 / MW[pollutant])
                            if pollutant == "co":
                                pollutant_data[pollutant] = ppb / 1000.0  # ppm for CO
                            elif pollutant == "o3":
                                pollutant_data[pollutant] = ppb / 1000.0  # ppm for O3 8h
                            else:
                                pollutant_data[pollutant] = ppb  # ppb for NO2, SO2
                        except Exception:
                            pollutant_data[pollutant] = np.nan
                    else:
                        pollutant_data[pollutant] = float(value)

            # Calculate AQI using the EPA method
            aqi = self.overall_aqi(
                pm2_5=pollutant_data.get("pm2_5"),
                pm10=pollutant_data.get("pm10"),
                o3_8h=pollutant_data.get("o3"),
                no2=pollutant_data.get("no2"), no2_unit="ppb",
                co=pollutant_data.get("co"),
                so2=pollutant_data.get("so2"), so2_unit="ppb",
            )
            aqi_values.append(aqi)

        return pd.Series(aqi_values, index=df.index)
    
    def validate_aqi_calculation(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Backwards-compatible validation focused on PM endpoints used previously.
        Returns dict mapping pollutant -> list of (calculated, expected).
        """
        test_cases = {
            "pm2_5": [(12.0, 50), (35.4, 100), (55.4, 150), (150.4, 200), (250.4, 300), (500.4, 500)],
            "pm10": [(54, 50), (154, 100), (254, 150), (354, 200), (424, 300), (604, 500)],
        }
        results: Dict[str, List[Tuple[float, float]]] = {}
        for pol, cases in test_cases.items():
            vals: List[Tuple[float, float]] = []
            for conc, expected in cases:
                if pol == "pm2_5":
                    calc = self.si_pm25(conc)
                elif pol == "pm10":
                    calc = self.si_pm10(conc)
                else:
                    calc = np.nan
                vals.append((calc, expected))
            results[pol] = vals
        return results

    def validate_aqi_range(self, aqi_values: pd.Series) -> Dict[str, any]:
        """
        Validate that calculated AQI values fall within reasonable EPA ranges.
        
        Args:
            aqi_values: Series of calculated AQI values
            
        Returns:
            Dictionary with validation results and statistics
        """
        if aqi_values.empty:
            return {"valid": False, "error": "Empty AQI values"}
        
        # EPA AQI ranges
        min_valid_aqi = 0
        max_valid_aqi = 500
        
        # Check for reasonable ranges
        min_aqi = aqi_values.min()
        max_aqi = aqi_values.max()
        mean_aqi = aqi_values.mean()
        std_aqi = aqi_values.std()
        
        # Validation checks
        valid_min = min_aqi >= min_valid_aqi
        valid_max = max_aqi <= max_valid_aqi
        reasonable_mean = 50 <= mean_aqi <= 300  # Most cities fall in this range
        reasonable_std = std_aqi <= 100  # AQI shouldn't vary too wildly
        
        # Check for suspicious patterns
        suspicious_perfect = aqi_values.nunique() < 10  # Too few unique values
        suspicious_round = (aqi_values % 1 == 0).sum() / len(aqi_values) > 0.8  # Too many round numbers
        
        validation_result = {
            "valid": all([valid_min, valid_max, reasonable_mean, reasonable_std]),
            "statistics": {
                "count": len(aqi_values),
                "min": min_aqi,
                "max": max_aqi, 
                "mean": mean_aqi,
                "std": std_aqi,
                "unique_values": aqi_values.nunique()
            },
            "checks": {
                "min_in_range": valid_min,
                "max_in_range": valid_max,
                "mean_reasonable": reasonable_mean,
                "std_reasonable": reasonable_std,
                "suspicious_perfect": suspicious_perfect,
                "suspicious_round": suspicious_round
            },
            "warnings": []
        }
        
        # Add warnings for suspicious patterns
        if suspicious_perfect:
            validation_result["warnings"].append("Too few unique AQI values - possible data leakage")
        if suspicious_round:
            validation_result["warnings"].append("Too many round AQI values - possible calculation error")
        if not valid_min or not valid_max:
            validation_result["warnings"].append("AQI values outside EPA range (0-500)")
        if not reasonable_mean:
            validation_result["warnings"].append(f"Mean AQI ({mean_aqi:.1f}) outside typical city range (50-300)")
        
        return validation_result


# ------------------
# Demo / CLI
# ------------------

def _demo():
    calc = EPAAQICalculator(clamp_to_500=True)

    print("🧪 Known-point validation (calculated vs expected equals):")
    val = calc.validate_known_points()
    for k, pairs in val.items():
        ok = sum(1 for c, e in pairs if abs(c - e) < 1)
        print(f"  {k:6s}: {ok}/{len(pairs)} correct")

    # Scenarios from user (values are already in proper units: PM in μg/m³, gases in ppm)
    scenarios = [
        {'pm2_5': 15.2, 'pm10': 45.3, 'o3_8h': 0.065, 'no2': 0.080, 'co': 7.2,  'so2': 0.050},
        {'pm2_5': 45.0, 'pm10': 120.0,'o3_8h': 0.085, 'no2': 0.150, 'co': 12.0, 'so2': 0.100},
        {'pm2_5': 120.0,'pm10': 250.0,'o3_8h': 0.120, 'no2': 0.400, 'co': 20.0, 'so2': 0.300},
    ]
    for i, s in enumerate(scenarios, 1):
        overall = calc.overall_aqi(
            pm2_5=s['pm2_5'],
            pm10=s['pm10'],
            o3_8h=s['o3_8h'],
            no2=s['no2'], no2_unit="ppm",
            co=s['co'],
            so2=s['so2'], so2_unit="ppm",
        )
        print(f"\nScenario {i}: Overall AQI = {overall} ({aqi_category(overall)})")

    # Example DataFrame usage
    df = pd.DataFrame([
        {"pm2_5": 15.2, "pm10": 45.3, "o3": 0.065, "no2": 0.080, "co": 7.2,  "so2": 0.050},
        {"pm2_5": 45.0, "pm10": 120.0,"o3": 0.085, "no2": 0.150, "co": 12.0, "so2": 0.100},
        {"pm2_5": 120.0,"pm10": 250.0,"o3": 0.120, "no2": 0.400, "co": 20.0, "so2": 0.300},
    ])
    df_out = calc.compute_aqi_for_dataframe(
        df,
        colmap=None,           # uses df columns as-is
        no2_unit="ppm",
        so2_unit="ppm",
        o3_columns=("o3", None)  # only 8-hr O3 available
    )
    print("\nDataFrame with AQI columns:")
    print(df_out)

if __name__ == "__main__":
    _demo()


