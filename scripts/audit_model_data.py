#!/usr/bin/env python3
"""
Read-only audit of football prediction training data.

Scans JSON files under data/ and reports:
- recognized match records and IDs
- likely score/date coverage
- xG-, shot-, and corner-related fields
- IDs that occur in multiple files
- common non-empty JSON field paths with sample values

It does not modify any source data. Field detection is heuristic; review the
reported field paths before using any metric to train a model.
"""
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "model-data-audit"
EXCLUDE_NAMES = {"backtest.json", "dataset_audit.json", "model_data_audit.json"}
ID_KEYS = ("id", "match_id", "matchId")
DATE_KEYS = ("utcDate", "date", "start_time", "startTime", "kickoff", "kickoff_time")


def nonempty(value):
    return value not in (None, "", [], {})


def first_value(obj, keys):
    if not isinstance(obj, dict):
        return None
    for key in keys:
        if nonempty(obj.get(key)):
            return obj[key]
    return None


def walk_fields(value, prefix=""):
    """Yield (JSON path, value); sample only the first two list entries."""
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from walk_fields(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value[:2]):
            yield from walk_fields(child, f"{prefix}[{index}]")


def extract_records(payload):
    """Support common list roots and keyed match stores."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    found = []
    for key in ("matches", "data", "results", "history", "events"):
        value = payload.get(key)
        if isinstance(value, list):
            found.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            found.extend(item for item in value.values() if isinstance(item, dict))

    if first_value(payload, ID_KEYS):
        found.append(payload)

    # Deduplicate records within this file when they have the same ID.
    output, seen = [], set()
    for item in found:
        match_id = first_value(item, ID_KEYS)
        if match_id is None and isinstance(item.get("match"), dict):
            match_id = first_value(item["match"], ID_KEYS)
        marker = str(match_id) if match_id is not None else None
        if marker is None or marker not in seen:
            output.append(item)
            if marker is not None:
                seen.add(marker)
    return output


def get_record_id(record):
    nested = record.get("match") if isinstance(record.get("match"), dict) else {}
    return first_value(record, ID_KEYS) or first_value(nested, ID_KEYS)


def has_recognized_score(record):
    nested = record.get("match") if isinstance(record.get("match"), dict) else {}
    for obj in (record, nested):
        score = obj.get("score")
        if isinstance(score, dict) and any(
            score.get(key) is not None
            for key in ("home", "away", "home_score", "away_score", "fullTime", "full_time")
        ):
            return True
        if obj.get("score_home") is not None and obj.get("score_away") is not None:
            return True
        if obj.get("home_score") is not None and obj.get("away_score") is not None:
            return True
    return False


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    files = sorted(
        path for path in DATA_DIR.rglob("*.json")
        if path.name not in EXCLUDE_NAMES
        and OUTPUT_DIR.name not in path.parts
    )
    global_ids = defaultdict(set)
    file_rows, parse_errors = [], []
    field_counts, field_samples = Counter(), defaultdict(list)

    for path in files:
        relative = str(path.relative_to(ROOT))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append({"file": relative, "error": str(exc)})
            continue

        records = extract_records(payload)
        local_ids = set()
        counts = Counter()

        for record in records:
            match_id = get_record_id(record)
            if match_id is not None:
                marker = str(match_id)
                local_ids.add(marker)
                global_ids[marker].add(relative)

            counts["scores"] += bool(has_recognized_score(record))
            nested = record.get("match") if isinstance(record.get("match"), dict) else {}
            counts["dates"] += bool(
                first_value(record, DATE_KEYS) or first_value(nested, DATE_KEYS)
            )

            field_list = list(walk_fields(record))
            lowered = [(path.lower(), value) for path, value in field_list]
            metric_terms = {
                "xg_like": ("xg", "expected_goals", "expectedgoals", "xga"),
                "shots_like": ("shot", "shots", "on_target", "ontarget"),
                "corners_like": ("corner", "corners"),
            }
            for metric, terms in metric_terms.items():
                counts[metric] += any(
                    any(term in field_path for term in terms) and nonempty(value)
                    for field_path, value in lowered
                )

            for field_path, value in field_list:
                if nonempty(value):
                    field_counts[field_path] += 1
                    if len(field_samples[field_path]) < 2:
                        field_samples[field_path].append(str(value)[:100])

        file_rows.append({
            "file": relative,
            "bytes": path.stat().st_size,
            "records": len(records),
            "unique_ids": len(local_ids),
            "scores": counts["scores"],
            "dates": counts["dates"],
            "xg_like": counts["xg_like"],
            "shots_like": counts["shots_like"],
            "corners_like": counts["corners_like"],
        })

    duplicates = {
        match_id: sorted(paths)
        for match_id, paths in global_ids.items()
        if len(paths) > 1
    }
    field_rows = [
        {"path": path, "occurrences": count, "samples": field_samples[path]}
        for path, count in field_counts.most_common(250)
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_scanned": len(files),
        "parse_errors": parse_errors,
        "unique_match_ids": len(global_ids),
        "ids_in_multiple_files": len(duplicates),
        "duplicate_examples": dict(list(duplicates.items())[:100]),
        "files": file_rows,
        "common_nonempty_field_paths": field_rows,
        "notes": [
            "Counts across files are not deduplicated.",
            "xG/shots/corners detection is heuristic and needs field-path review.",
            "The audit is read-only and does not alter source JSON files.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "model_data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Model training data audit", "",
        f"- Generated UTC: {report['generated_at_utc']}",
        f"- JSON files scanned: {len(files)}",
        f"- Unique match IDs: {len(global_ids)}",
        f"- IDs appearing in multiple files: {len(duplicates)}",
        f"- Parse errors: {len(parse_errors)}", "",
        "## Coverage by file", "",
        "| File | Records | IDs | Scores | xG-like | Shots-like | Corners-like |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in file_rows:
        lines.append(
            f"| {row['file']} | {row['records']} | {row['unique_ids']} | "
            f"{row['scores']} | {row['xg_like']} | {row['shots_like']} | "
            f"{row['corners_like']} |"
        )
    lines += [
        "", "## Common non-empty JSON field paths", "",
        "| Field path | Occurrences | Example value |",
        "|---|---:|---|",
    ]
    for row in field_rows[:120]:
        sample = (row["samples"] or [""])[0].replace("|", "/")
        lines.append(f"| `{row['path']}` | {row['occurrences']} | {sample} |")

    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
