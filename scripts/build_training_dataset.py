#!/usr/bin/env python3
"""
Build leakage-safe rolling features from PitchAPI match history.

Run from the repository root:
    python scripts/build_training_dataset.py

Reads JSON files under data/pitchapi_history/ and data/pitchapi_stats.json.
Writes:
    data/training_dataset.csv
    data/training_dataset_report.json

IMPORTANT:
- A match's own xG/shots/stats are labels/diagnostics for that match, never
  predictors for that same match.
- Predictors are rolling summaries from each team's earlier matches only.
- This is a first-pass feature builder. Review the report before model training.
"""
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY_DIR = DATA / "pitchapi_history"
STATS_FILE = DATA / "pitchapi_stats.json"
CSV_OUT = DATA / "training_dataset.csv"
REPORT_OUT = DATA / "training_dataset_report.json"
WINDOW = 5


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: Could not parse {path}: {exc}")
        return None


def iter_records(payload):
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for key in ("matches", "data", "results", "history", "events"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
            if isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, dict):
                        yield item
                return
        if isinstance(payload.get("match"), dict):
            yield payload


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def numeric(value):
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def match_identifier(record):
    match = record.get("match") if isinstance(record.get("match"), dict) else {}
    for obj in (match, record):
        for key in ("id", "match_id", "matchId"):
            if obj.get(key) not in (None, ""):
                return str(obj[key])
    return None


def team_name(record, side):
    match = record.get("match") if isinstance(record.get("match"), dict) else {}
    team = match.get(f"{side}_team")
    if isinstance(team, dict) and team.get("name"):
        return str(team["name"])
    legacy = record.get(f"{side}Team")
    if isinstance(legacy, dict) and legacy.get("name"):
        return str(legacy["name"])
    return None


def match_date(record):
    match = record.get("match") if isinstance(record.get("match"), dict) else {}
    for obj in (match, record):
        for key in ("time_utc", "utcDate", "date", "start_time", "startTime", "kickoff"):
            parsed = parse_date(obj.get(key))
            if parsed:
                return parsed
    return None


def score_pair(record):
    match = record.get("match") if isinstance(record.get("match"), dict) else {}
    for obj in (match, record):
        home = numeric(obj.get("score_home"))
        away = numeric(obj.get("score_away"))
        if home is not None and away is not None:
            return int(home), int(away)
        score = obj.get("score")
        if isinstance(score, dict):
            full = score.get("fullTime") or score.get("full_time") or score
            if isinstance(full, dict):
                home = numeric(full.get("home"))
                away = numeric(full.get("away"))
                if home is not None and away is not None:
                    return int(home), int(away)
    return None


def extract_stat_items(record):
    """Map normalized stat keys to (home, away) numeric values."""
    result = {}
    stats = record.get("stats")
    if not isinstance(stats, dict):
        return result
    periods = stats.get("periods")
    if not isinstance(periods, list):
        return result
    period = next((p for p in periods if isinstance(p, dict) and str(p.get("period", "")).lower() in ("all", "fulltime", "full_time")), None)
    if period is None and periods:
        period = periods[0] if isinstance(periods[0], dict) else None
    if not period:
        return result
    for group in period.get("groups", []) if isinstance(period.get("groups"), list) else []:
        if not isinstance(group, dict):
            continue
        for item in group.get("items", []) if isinstance(group.get("items"), list) else []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or item.get("title") or "").strip().lower()
            key = key.replace(" ", "_").replace("-", "_")
            home, away = numeric(item.get("home")), numeric(item.get("away"))
            if key and home is not None and away is not None:
                result[key] = (home, away)
    return result


def extract_xg_shots(record):
    xg = {"home": 0.0, "away": 0.0}
    shots = {"home": 0, "away": 0}
    on_target = {"home": 0, "away": 0}
    found_xg = False
    shot_data = record.get("shots")
    if not isinstance(shot_data, dict):
        return {"home_xg": None, "away_xg": None, "home_shots": None, "away_shots": None,
                "home_shots_on_target": None, "away_shots_on_target": None}
    for period in shot_data.get("periods", []) if isinstance(shot_data.get("periods"), list) else []:
        if not isinstance(period, dict):
            continue
        for shot in period.get("shots", []) if isinstance(period.get("shots"), list) else []:
            if not isinstance(shot, dict):
                continue
            side = None
            team_id = shot.get("team_id")
            match = record.get("match") if isinstance(record.get("match"), dict) else {}
            for candidate in ("home", "away"):
                team = match.get(f"{candidate}_team")
                if isinstance(team, dict) and team.get("id") is not None and str(team["id"]) == str(team_id):
                    side = candidate
                    break
            if side is None:
                continue
            shots[side] += 1
            if shot.get("is_on_target") is True:
                on_target[side] += 1
            value = numeric(shot.get("expected_goals"))
            if value is not None:
                xg[side] += value
                found_xg = True
    return {
        "home_xg": round(xg["home"], 4) if found_xg else None,
        "away_xg": round(xg["away"], 4) if found_xg else None,
        "home_shots": shots["home"],
        "away_shots": shots["away"],
        "home_shots_on_target": on_target["home"],
        "away_shots_on_target": on_target["away"],
    }


def extract_corners(stats):
    # PitchAPI keys may vary by competition/provider; accept common spellings.
    candidates = ("corner_kicks", "corners", "corner", "total_corners")
    for key in candidates:
        if key in stats:
            return stats[key]
    for key, values in stats.items():
        if "corner" in key:
            return values
    return None


def main():
    if not HISTORY_DIR.exists():
        raise SystemExit(f"Missing directory: {HISTORY_DIR}")

    by_id = {}
    files_read = 0
    raw_records = 0
    parse_errors = []
    for path in sorted(HISTORY_DIR.rglob("*.json")):
        payload = read_json(path)
        if payload is None:
            parse_errors.append(str(path.relative_to(ROOT)))
            continue
        files_read += 1
        for record in iter_records(payload):
            raw_records += 1
            mid = match_identifier(record)
            date = match_date(record)
            home, away = team_name(record, "home"), team_name(record, "away")
            score = score_pair(record)
            if not mid or not date or not home or not away or not score:
                continue
            # Prefer record with more nested stats/shots when duplicate IDs occur.
            quality = int(bool(record.get("stats"))) + int(bool(record.get("shots")))
            old = by_id.get(mid)
            if old is None or quality > old["_quality"]:
                stat_items = extract_stat_items(record)
                xg_shots = extract_xg_shots(record)
                by_id[mid] = {
                    "match_id": mid, "date": date, "home_team": home, "away_team": away,
                    "home_goals": score[0], "away_goals": score[1],
                    "competition": str((record.get("competition") or {}).get("name", "")
                                       if isinstance(record.get("competition"), dict)
                                       else record.get("competition") or ""),
                    **xg_shots,
                    "home_corners": (extract_corners(stat_items) or (None, None))[0]
                        if extract_corners(stat_items) else None,
                    "away_corners": (extract_corners(stat_items) or (None, None))[1]
                        if extract_corners(stat_items) else None,
                    "_stats": stat_items, "_quality": quality,
                }

    matches = sorted(by_id.values(), key=lambda row: row["date"])
    history = defaultdict(list)
    output_rows = []
    feature_coverage = {
        "matches": len(matches),
        "matches_with_xg": sum(m["home_xg"] is not None and m["away_xg"] is not None for m in matches),
        "matches_with_shots": sum(m["home_shots"] is not None and m["away_shots"] is not None for m in matches),
        "matches_with_corners": sum(m["home_corners"] is not None and m["away_corners"] is not None for m in matches),
    }

    for match in matches:
        home_hist = history[match["home_team"]]
        away_hist = history[match["away_team"]]
        row = {
            "match_id": match["match_id"],
            "date_utc": match["date"].isoformat(),
            "competition": match["competition"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "home_goals": match["home_goals"],
            "away_goals": match["away_goals"],
            "target_home_win": int(match["home_goals"] > match["away_goals"]),
            "target_over_1_5": int(match["home_goals"] + match["away_goals"] >= 2),
            "target_total_corners": None if match["home_corners"] is None or match["away_corners"] is None
                                    else match["home_corners"] + match["away_corners"],
            "match_home_xg_label": match["home_xg"],
            "match_away_xg_label": match["away_xg"],
            "match_home_shots_label": match["home_shots"],
            "match_away_shots_label": match["away_shots"],
            "match_home_shots_on_target_label": match["home_shots_on_target"],
            "match_away_shots_on_target_label": match["away_shots_on_target"],
            "match_home_corners_label": match["home_corners"],
            "match_away_corners_label": match["away_corners"],
            "home_prior_matches": len(home_hist),
            "away_prior_matches": len(away_hist),
        }
        for side, team_hist in (("home", home_hist), ("away", away_hist)):
            recent = team_hist[-WINDOW:]
            for metric in ("goals_for", "goals_against", "xg_for", "xg_against",
                           "shots_for", "shots_against", "shots_on_target_for",
                           "shots_on_target_against", "corners_for", "corners_against"):
                vals = [h.get(metric) for h in recent]
                vals = [v for v in vals if v is not None]
                row[f"{side}_last{WINDOW}_{metric}_avg"] = round(sum(vals) / len(vals), 4) if vals else None
            row[f"{side}_last{WINDOW}_n_with_xg"] = sum(h.get("xg_for") is not None for h in recent)
            row[f"{side}_last{WINDOW}_n_with_shots"] = sum(h.get("shots_for") is not None for h in recent)
        output_rows.append(row)

        # Only after emitting the prediction row do we add this match to team history.
        home_stats, away_stats = match["_stats"], match["_stats"]
        home_xg, away_xg = match["home_xg"], match["away_xg"]
        history[match["home_team"]].append({
            "goals_for": match["home_goals"], "goals_against": match["away_goals"],
            "xg_for": home_xg, "xg_against": away_xg,
            "shots_for": match["home_shots"], "shots_against": match["away_shots"],
            "shots_on_target_for": match["home_shots_on_target"],
            "shots_on_target_against": match["away_shots_on_target"],
            "corners_for": match["home_corners"], "corners_against": match["away_corners"],
        })
        history[match["away_team"]].append({
            "goals_for": match["away_goals"], "goals_against": match["home_goals"],
            "xg_for": away_xg, "xg_against": home_xg,
            "shots_for": match["away_shots"], "shots_against": match["home_shots"],
            "shots_on_target_for": match["away_shots_on_target"],
            "shots_on_target_against": match["home_shots_on_target"],
            "corners_for": match["away_corners"], "corners_against": match["home_corners"],
        })

    if not output_rows:
        raise SystemExit("No valid finished matches found. Review source JSON schema and field names.")

    fieldnames = list(output_rows[0].keys())
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(HISTORY_DIR.relative_to(ROOT)),
        "source_files_read": files_read,
        "raw_records_seen": raw_records,
        "unique_finished_matches_used": len(matches),
        "duplicate_match_ids_collapsed": max(0, raw_records - len(by_id)),
        "parse_errors": parse_errors,
        "coverage": feature_coverage,
        "output_csv": str(CSV_OUT.relative_to(ROOT)),
        "leakage_protection": "Rolling team features are calculated before each match is added to history.",
        "limitations": [
            "Only files under data/pitchapi_history are used in this first pass.",
            "Corners depend on a recognizable corner stat key in stats.periods[].groups[].items[].key.",
            "Team names may differ across competitions/seasons; inspect before merging team histories.",
            "xG/shots/corners coverage is not assumed complete; missing values remain blank.",
            "Do not train a model until report coverage and feature columns have been reviewed."
        ]
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nCreated: {CSV_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
