"""
Check dataset coverage at each stage:
  1) raw: scenario_*.parquet count (same as convert_argoverse2 input)
  2) converted: ScenarioNet converted count (after python -m scenarionet.convert_argoverse2)
  3) loaded: actually loaded by UniTraj (after object_type / center object filter)

Use --raw_data_path to verify conversion didn't omit any (raw vs converted).
"""
import os
import pickle
import argparse
from pathlib import Path


def count_raw_scenarios(raw_split_path):
    """Same logic as scenarionet get_av2_scenarios: count scenario_*.parquet files."""
    if not os.path.isdir(raw_split_path):
        return None
    count = 0
    for p in Path(raw_split_path).rglob("*.parquet"):
        if p.name.startswith("scenario_"):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Check dataset coverage: raw (parquet) → converted → loaded."
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default="/home/hsjung-larr/workspace/motion_forecasting/argoverse2",
        help="Path to converted argoverse2 root (containing train/, val/, test/)",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="./cache",
        help="Path to UniTraj cache root (same as config cache_path)",
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="/home/hsjung-larr/workspace/motion_forecasting/data",
        help="Path to raw data root (containing train/, val/, test/ with scenario_*.parquet). "
             "If set, compares raw count vs converted to check conversion omissions.",
    )
    args = parser.parse_args()

    try:
        from scenarionet.common_utils import read_dataset_summary
    except ImportError:
        print("Install scenarionet first: pip install scenarionet")
        return

    database_path = os.path.abspath(args.database_path)
    cache_path = os.path.abspath(args.cache_path)
    raw_data_path = os.path.abspath(args.raw_data_path) if args.raw_data_path else None

    print(f"Database path (converted): {database_path}")
    print(f"Cache path:                {cache_path}")
    if raw_data_path:
        print(f"Raw data path:             {raw_data_path}")
    print()

    header = f"{'split':5} | {'raw':6} | {'converted':10} | {'loaded':6} | notes"
    print(header)
    print("-" * len(header))

    for split in ("train", "val", "test"):
        # Raw: parquet count (convert_argoverse2 input)
        if raw_data_path:
            raw_split = os.path.join(raw_data_path, split)
            raw_count = count_raw_scenarios(raw_split)
        else:
            raw_count = None

        # Converted: ScenarioNet output
        split_path = os.path.join(database_path, split)
        if not os.path.isdir(split_path):
            print(f"{split:5} | {'-':6} | (path not found)")
            continue
        try:
            summary, scenario_list, mapping = read_dataset_summary(split_path)
            total_converted = len(scenario_list)
        except Exception as e:
            print(f"{split:5} | {str(raw_count) if raw_count is not None else '-':>6} | error: {e}")
            continue

        # Loaded: UniTraj after filter
        dataset_name = os.path.basename(database_path)
        cache_file = os.path.join(cache_path, split, dataset_name, "file_list.pkl")
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                file_list = pickle.load(f)
            loaded = len(file_list)
        else:
            loaded = None

        raw_s = str(raw_count) if raw_count is not None else "-"
        loaded_s = str(loaded) if loaded is not None else "(no cache)"
        notes = []
        if raw_count is not None and total_converted != raw_count:
            notes.append(f"conversion: {raw_count - total_converted} omitted")
        if loaded is not None and total_converted != loaded:
            notes.append(f"UniTraj filter: {total_converted - loaded} removed")
        note_s = "; ".join(notes) if notes else ""
        print(f"{split:5} | {raw_s:>6} | {total_converted:>10} | {loaded_s:>6} | {note_s}")

    print()
    print("raw       = scenario_*.parquet count (input to convert_argoverse2)")
    print("converted = ScenarioNet converted count (output of convert_argoverse2)")
    print("loaded    = UniTraj loaded (after object_type / center object filter)")

if __name__ == "__main__":
    main()
