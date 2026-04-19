from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
RAW_ZHVI_PATH = ROOT / "data" / "raw" / "County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
EXTERNAL_FEATURES_PATH = ROOT / "data" / "processed" / "external_features.csv"
FRB_MACRO_OUTPUT_PATH = ROOT / "data" / "processed" / "frb_macro_features.csv"
FRB_METADATA_OUTPUT_PATH = ROOT / "data" / "processed" / "frb_series_metadata.csv"
PANEL_OUTPUT_PATH = ROOT / "data" / "processed" / "panel_dataset.csv"

ID_COLUMNS = [
    "RegionID",
    "SizeRank",
    "RegionName",
    "RegionType",
    "StateName",
    "State",
    "Metro",
    "StateCodeFIPS",
    "MunicipalCodeFIPS",
]

FRB_SERIES_NAME_OVERRIDES = {
    ("FRB_G17.csv", "IP.B50001.S"): "industrial_production_total_sa",
    ("FRB_G17.csv", "IP.GMF.S"): "industrial_production_manufacturing_sa",
    ("FRB_G19.csv", "DTCTL.M"): "consumer_credit_total_sa",
    ("FRB_G19.csv", "DTCTL_@%A_BA.M"): "consumer_credit_growth_annualized",
    ("FRB_G20.csv", "RIELPCFAN_N.M"): "new_car_loan_rate_finance_companies",
    ("FRB_G20.csv", "RIELPCFAU_N.M"): "used_car_loan_rate_finance_companies",
    ("FRB_H10.csv", "JRXWTFB_N.M"): "dollar_index_broad",
    ("FRB_H15.csv", "RIFSPFF_N.M"): "fed_funds_rate",
    ("FRB_H15.csv", "RIFSPBLP_N.M"): "prime_rate",
    ("FRB_H15.csv", "RIFLGFCM03_N.M"): "treasury_3m",
    ("FRB_H15.csv", "RIFLGFCY02_N.M"): "treasury_2y",
    ("FRB_H15.csv", "RIFLGFCY10_N.M"): "treasury_10y",
    ("FRB_H15.csv", "RIFLGFCY30_N.M"): "treasury_30y",
    ("FRB_H3.csv", "RESTR14A_N.M"): "depository_reserves_total",
    ("FRB_H3.csv", "RESMO14A_N.M"): "monetary_base_total",
}


def normalize_month_end(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    return timestamps.dt.to_period("M").dt.to_timestamp("M")


def sanitize_column_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unnamed_series"


def load_frb_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_rows = pd.read_csv(path, header=None, nrows=6, dtype="string")
    frame = pd.read_csv(path, skiprows=5)

    if "Time Period" not in frame.columns:
        raise ValueError(f"Missing 'Time Period' header in {path.name}.")

    renamed_columns: dict[str, str] = {}
    series_metadata: list[dict[str, str]] = []

    for position, original_name in enumerate(frame.columns[1:], start=1):
        column_name = FRB_SERIES_NAME_OVERRIDES.get(
            (path.name, original_name),
            f"{sanitize_column_name(path.stem)}_{sanitize_column_name(original_name)}",
        )
        renamed_columns[original_name] = column_name
        series_metadata.append(
            {
                "source_file": path.name,
                "series_code": original_name,
                "column_name": column_name,
                "description": metadata_rows.iat[0, position],
                "unit": metadata_rows.iat[1, position],
                "multiplier": metadata_rows.iat[2, position],
                "currency": metadata_rows.iat[3, position],
                "unique_identifier": metadata_rows.iat[4, position],
            }
        )

    frame = frame.rename(columns={"Time Period": "date", **renamed_columns})
    frame["date"] = normalize_month_end(frame["date"])

    feature_columns = list(renamed_columns.values())
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame[["date"] + feature_columns].sort_values("date").reset_index(drop=True)

    return frame, pd.DataFrame(series_metadata)


def load_frb_macro_features(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = FRB_MACRO_OUTPUT_PATH,
    metadata_output_path: Path = FRB_METADATA_OUTPUT_PATH,
) -> pd.DataFrame | None:
    frb_paths = sorted(raw_dir.glob("FRB_*.csv"))
    if not frb_paths:
        return None

    macro_frames = []
    metadata_frames = []

    for path in frb_paths:
        features, metadata = load_frb_file(path)
        macro_frames.append(features)
        metadata_frames.append(metadata)

    macro_features = macro_frames[0]
    for frame in macro_frames[1:]:
        macro_features = macro_features.merge(frame, on="date", how="outer", validate="one_to_one")

    macro_features = macro_features.sort_values("date").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro_features.to_csv(output_path, index=False)
    pd.concat(metadata_frames, ignore_index=True).to_csv(metadata_output_path, index=False)
    return macro_features


def load_zhvi_panel(path: Path = RAW_ZHVI_PATH) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "StateCodeFIPS": "string",
            "MunicipalCodeFIPS": "string",
        },
    )

    date_columns = [column for column in frame.columns if column not in ID_COLUMNS]

    panel = frame.melt(
        id_vars=ID_COLUMNS,
        value_vars=date_columns,
        var_name="date",
        value_name="zhvi",
    )

    panel = panel.rename(
        columns={
            "RegionID": "region_id",
            "SizeRank": "size_rank",
            "RegionName": "county_name",
            "RegionType": "region_type",
            "StateName": "state_name",
            "State": "state",
            "Metro": "metro",
            "StateCodeFIPS": "state_fips",
            "MunicipalCodeFIPS": "county_code",
        }
    )

    panel["county_fips"] = (
        panel["state_fips"].fillna("").str.zfill(2)
        + panel["county_code"].fillna("").str.zfill(3)
    )
    panel["date"] = normalize_month_end(panel["date"])
    panel["zhvi"] = pd.to_numeric(panel["zhvi"], errors="coerce")

    panel = panel.loc[panel["region_type"].str.lower() == "county"].copy()
    panel = panel.sort_values(["county_fips", "date"]).reset_index(drop=True)

    return panel


def load_external_features(path: Path = EXTERNAL_FEATURES_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None

    frame = pd.read_csv(path, dtype={"county_fips": "string"})
    frame["county_fips"] = frame["county_fips"].str.zfill(5)
    frame["date"] = normalize_month_end(frame["date"])
    return frame


def build_panel_dataset(
    raw_zhvi_path: Path = RAW_ZHVI_PATH,
    external_features_path: Path = EXTERNAL_FEATURES_PATH,
    output_path: Path = PANEL_OUTPUT_PATH,
) -> pd.DataFrame:
    panel = load_zhvi_panel(raw_zhvi_path)
    macro_features = load_frb_macro_features(raw_dir=raw_zhvi_path.parent)
    external_features = load_external_features(external_features_path)

    if macro_features is not None:
        panel = panel.merge(
            macro_features,
            on="date",
            how="left",
            validate="many_to_one",
        )

    if external_features is not None:
        panel = panel.merge(
            external_features,
            on=["county_fips", "date"],
            how="left",
            validate="one_to_one",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    return panel
