#!/usr/bin/env python3
"""Chronological comparison of the current and a less home-biased model.

Each match is predicted only from matches dated before it. The script keeps
the current model unchanged as a benchmark and tests one candidate variant.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "data" / "history.json"
OUTPUT_PATH = ROOT / "data" / "backtest.json"
MIN_TEAM_MATCHES = 3
MAX_GOALS = 8
EPS = 1e-12
OUTCOMES = ("1", "X", "2")


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def finished_match(m):
    score = (m.get("score") or {}).get("fullTime") or {}
    return (
        m.get("status") == "FINISHED"
        and isinstance(score.get("home"), (int, float))
        and isinstance(score.get("away"), (int, float))
        and m.get("homeTeam", {}).get("name")
        and m.get("awayTeam", {}).get("name")
        and parse_date(m.get("utcDate"))
    )


def poisson_probs(lam):
    vals = [math.exp(-lam)]
    for k in range(1, MAX_GOALS + 1):
        vals.append(vals[-1] * lam / k)
    return vals


def team_form(prior, team, limit=5, home_only=False, away_only=False):
    relevant = []
    for m in prior:
        home = m["homeTeam"]["name"] == team
        away = m["awayTeam"]["name"] == team
        if not (home or away):
            continue
        if home_only and not home:
            continue
        if away_only and not away:
            continue
        relevant.append(m)
    relevant = relevant[-limit:]
    if not relevant:
        return None

    gf = ga = points = 0
    for m in relevant:
        score = m["score"]["fullTime"]
        is_home = m["homeTeam"]["name"] == team
        goals_for = score["home"] if is_home else score["away"]
        goals_against = score["away"] if is_home else score["home"]
        gf += goals_for
        ga += goals_against
        points += 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0

    n = len(relevant)
    return {"n": n, "gf": gf / n, "ga": ga / n, "points_per_match": points / n}


def probabilities(lam_home, lam_away):
    hp = poisson_probs(lam_home)
    ap = poisson_probs(lam_away)
    p1 = px = p2 = 0.0
    for i, pi in enumerate(hp):
        for j, pj in enumerate(ap):
            probability = pi * pj
            if i > j:
                p1 += probability
            elif i == j:
                px += probability
            else:
                p2 += probability
    total = p1 + px + p2
    return {"1": p1 / total, "X": px / total, "2": p2 / total}


def predict(prior, match, variant="current"):
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    home_all = team_form(prior, home, limit=10, home_only=True)
    away_all = team_form(prior, away, limit=10, away_only=True)
    home_recent = team_form(prior, home, limit=5)
    away_recent = team_form(prior, away, limit=5)

    if (not home_recent or not away_recent
            or home_recent["n"] < MIN_TEAM_MATCHES
            or away_recent["n"] < MIN_TEAM_MATCHES):
        return None

    h_attack = (home_all or home_recent)["gf"]
    h_concede = (home_all or home_recent)["ga"]
    a_attack = (away_all or away_recent)["gf"]
    a_concede = (away_all or away_recent)["ga"]
    form_h = home_recent["gf"] - home_recent["ga"]
    form_a = away_recent["gf"] - away_recent["ga"]

    if variant == "current":
        # Existing formula, retained as the comparison benchmark.
        lam_home = max(0.2, min(3.8, 0.52 * h_attack + 0.48 * a_concede + 0.12 + 0.06 * form_h))
        lam_away = max(0.2, min(3.8, 0.52 * a_attack + 0.48 * h_concede - 0.04 + 0.06 * form_a))
    else:
        # Experimental candidate: slightly less home weighting and a more
        # neutral baseline. This is a test candidate, not assumed to be better.
        lam_home = max(0.2, min(3.8, 0.48 * h_attack + 0.42 * a_concede + 0.08 + 0.04 * form_h))
        lam_away = max(0.2, min(3.8, 0.50 * a_attack + 0.50 * h_concede + 0.02 + 0.06 * form_a))

    probs = probabilities(lam_home, lam_away)
    return {
        **probs,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
    }


def actual_outcome(match):
    h = match["score"]["fullTime"]["home"]
    a = match["score"]["fullTime"]["away"]
    return "1" if h > a else "X" if h == a else "2"


def calculate_metrics(rows, variant):
    n = len(rows)
    if not n:
        return {}
    correct = sum(max(row[variant], key=row[variant].get) == row["actual"] for row in rows)
    accuracy = correct / n
    brier = sum(
        (row[variant][key] - (row["actual"] == key)) ** 2
        for row in rows for key in OUTCOMES
    ) / n
    log_loss = -sum(math.log(max(EPS, row[variant][row["actual"]])) for row in rows) / n
    predicted_counts = {
        key: sum(max(row[variant], key=row[variant].get) == key for row in rows)
        for key in OUTCOMES
    }
    return {
        "accuracy": round(accuracy, 5),
        "brier_score": round(brier, 5),
        "log_loss": round(log_loss, 5),
        "predicted_outcomes": predicted_counts,
    }


def main():
    if not HISTORY_PATH.exists():
        raise SystemExit(f"Missing historical data file: {HISTORY_PATH}")
    raw = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    matches = raw.get("matches", []) if isinstance(raw, dict) else raw
    matches = [m for m in matches if finished_match(m)]
    matches.sort(key=lambda m: parse_date(m["utcDate"]))

    rows = []
    skipped = 0
    prior = []
    for match in matches:
        current = predict(prior, match, "current")
        candidate = predict(prior, match, "candidate")
        if current is None or candidate is None:
            skipped += 1
        else:
            rows.append({
                "date": match["utcDate"],
                "competition": (match.get("competition") or {}).get("name", ""),
                "home": match["homeTeam"]["name"],
                "away": match["awayTeam"]["name"],
                "score_home": match["score"]["fullTime"]["home"],
                "score_away": match["score"]["fullTime"]["away"],
                "actual": actual_outcome(match),
                "current": current,
                "candidate": candidate,
            })
        prior.append(match)

    current_metrics = calculate_metrics(rows, "current")
    candidate_metrics = calculate_metrics(rows, "candidate")
    actual_counts = {key: sum(row["actual"] == key for row in rows) for key in OUTCOMES}
    baseline_accuracy = actual_counts["1"] / len(rows) if rows else None

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Chronological expanding-window comparison; each match uses only earlier matches.",
        "source_matches": len(matches),
        "tested_matches": len(rows),
        "skipped_early_matches": skipped,
        "metrics": {
            "current_model": current_metrics,
            "candidate_model": candidate_metrics,
            "always_home_win_baseline_accuracy": round(baseline_accuracy, 5) if baseline_accuracy is not None else None,
            "candidate_accuracy_change_vs_current": round(candidate_metrics["accuracy"] - current_metrics["accuracy"], 5) if rows else None,
            "candidate_brier_change_vs_current": round(candidate_metrics["brier_score"] - current_metrics["brier_score"], 5) if rows else None,
            "candidate_log_loss_change_vs_current": round(candidate_metrics["log_loss"] - current_metrics["log_loss"], 5) if rows else None,
            "actual_outcomes": actual_counts,
        },
        "predictions": [
            {
                "date": row["date"], "competition": row["competition"],
                "home": row["home"], "away": row["away"],
                "score_home": row["score_home"], "score_away": row["score_away"],
                "actual": row["actual"],
                "current": row["current"], "candidate": row["candidate"],
            }
            for row in rows
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("FOOTBALL MODEL COMPARISON")
    print(f"Historical matches: {len(matches)}")
    print(f"Predictions evaluated: {len(rows)}; skipped early matches: {skipped}")
    print(f"Current model: accuracy={current_metrics.get('accuracy', 0):.1%}, Brier={current_metrics.get('brier_score', 0):.4f}, log loss={current_metrics.get('log_loss', 0):.4f}")
    print(f"Candidate model: accuracy={candidate_metrics.get('accuracy', 0):.1%}, Brier={candidate_metrics.get('brier_score', 0):.4f}, log loss={candidate_metrics.get('log_loss', 0):.4f}")
    print(f"Always-home baseline: {baseline_accuracy:.1%}" if baseline_accuracy is not None else "Always-home baseline: unavailable")
    if rows:
        print(f"Candidate accuracy change: {candidate_metrics['accuracy'] - current_metrics['accuracy']:+.1%}")
        print(f"Candidate Brier change (negative is better): {candidate_metrics['brier_score'] - current_metrics['brier_score']:+.4f}")
        print(f"Candidate log-loss change (negative is better): {candidate_metrics['log_loss'] - current_metrics['log_loss']:+.4f}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
