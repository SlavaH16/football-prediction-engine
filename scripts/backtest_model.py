#!/usr/bin/env python3
"""Chronological backtest for the Football Prediction Engine.

Each historical match is predicted only from matches played earlier.
Writes data/backtest.json and prints model metrics plus a simple baseline.
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
    return {
        "n": n,
        "gf": gf / n,
        "ga": ga / n,
        "points_per_match": points / n,
    }


def predict(prior, match):
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    home_all = team_form(prior, home, limit=10, home_only=True)
    away_all = team_form(prior, away, limit=10, away_only=True)
    home_recent = team_form(prior, home, limit=5)
    away_recent = team_form(prior, away, limit=5)

    if (
        not home_recent
        or not away_recent
        or home_recent["n"] < MIN_TEAM_MATCHES
        or away_recent["n"] < MIN_TEAM_MATCHES
    ):
        return None

    h_attack = (home_all or home_recent)["gf"]
    h_concede = (home_all or home_recent)["ga"]
    a_attack = (away_all or away_recent)["gf"]
    a_concede = (away_all or away_recent)["ga"]
    form_h = home_recent["gf"] - home_recent["ga"]
    form_a = away_recent["gf"] - away_recent["ga"]

    lam_home = max(0.2, min(3.8, 0.52 * h_attack + 0.48 * a_concede + 0.12 + 0.06 * form_h))
    lam_away = max(0.2, min(3.8, 0.52 * a_attack + 0.48 * h_concede - 0.04 + 0.06 * form_a))

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
    return {"p1": p1 / total, "px": px / total, "p2": p2 / total,
            "lambda_home": lam_home, "lambda_away": lam_away}


def actual_outcome(match):
    h = match["score"]["fullTime"]["home"]
    a = match["score"]["fullTime"]["away"]
    return "1" if h > a else "X" if h == a else "2"


def main():
    if not HISTORY_PATH.exists():
        raise SystemExit(f"Missing historical data file: {HISTORY_PATH}")

    raw = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    matches = raw.get("matches", []) if isinstance(raw, dict) else raw
    matches = [m for m in matches if finished_match(m)]
    matches.sort(key=lambda m: parse_date(m["utcDate"]))

    predictions = []
    skipped = 0
    prior = []

    for match in matches:
        prediction = predict(prior, match)
        if prediction is None:
            skipped += 1
        else:
            actual = actual_outcome(match)
            probs = {"1": prediction["p1"], "X": prediction["px"], "2": prediction["p2"]}
            predictions.append({
                "date": match["utcDate"],
                "competition": (match.get("competition") or {}).get("name", ""),
                "home": match["homeTeam"]["name"],
                "away": match["awayTeam"]["name"],
                "score_home": match["score"]["fullTime"]["home"],
                "score_away": match["score"]["fullTime"]["away"],
                "actual": actual,
                "predicted": max(probs, key=probs.get),
                "p1": round(prediction["p1"], 6),
                "px": round(prediction["px"], 6),
                "p2": round(prediction["p2"], 6),
                "expected_goals_home": round(prediction["lambda_home"], 3),
                "expected_goals_away": round(prediction["lambda_away"], 3),
            })
        prior.append(match)

    n = len(predictions)
    if n:
        correct = sum(p["predicted"] == p["actual"] for p in predictions)
        accuracy = correct / n
        brier = sum(
            (p["p1"] - (p["actual"] == "1")) ** 2
            + (p["px"] - (p["actual"] == "X")) ** 2
            + (p["p2"] - (p["actual"] == "2")) ** 2
            for p in predictions
        ) / n
        log_loss = -sum(
            math.log(max(EPS, {"1": p["p1"], "X": p["px"], "2": p["p2"]}[p["actual"]]))
            for p in predictions
        ) / n
        actual_counts = {key: sum(p["actual"] == key for p in predictions) for key in ("1", "X", "2")}
        predicted_counts = {key: sum(p["predicted"] == key for p in predictions) for key in ("1", "X", "2")}

        # Baseline: predict a home win for every match.
        baseline_accuracy = actual_counts["1"] / n
        accuracy_lift = accuracy - baseline_accuracy
    else:
        accuracy = brier = log_loss = baseline_accuracy = accuracy_lift = None
        actual_counts = predicted_counts = {"1": 0, "X": 0, "2": 0}

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Chronological expanding-window backtest; each match is predicted only from earlier matches.",
        "source_matches": len(matches),
        "tested_matches": n,
        "skipped_early_matches": skipped,
        "metrics": {
            "accuracy": round(accuracy, 5) if accuracy is not None else None,
            "always_home_win_baseline_accuracy": round(baseline_accuracy, 5) if baseline_accuracy is not None else None,
            "accuracy_lift_vs_baseline": round(accuracy_lift, 5) if accuracy_lift is not None else None,
            "multiclass_brier_score": round(brier, 5) if brier is not None else None,
            "log_loss": round(log_loss, 5) if log_loss is not None else None,
            "actual_outcomes": actual_counts,
            "predicted_outcomes": predicted_counts,
        },
        "predictions": predictions,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("FOOTBALL MODEL BACKTEST")
    print(f"Historical matches after validation: {len(matches)}")
    print(f"Predictions evaluated: {n}; skipped early matches: {skipped}")
    if n:
        print(f"Model accuracy 1/X/2: {accuracy:.1%}")
        print(f"Baseline (always predict home win): {baseline_accuracy:.1%}")
        print(f"Accuracy difference vs baseline: {accuracy_lift:+.1%}")
        print(f"Brier score (lower is better): {brier:.4f}")
        print(f"Log loss (lower is better): {log_loss:.4f}")
        print(f"Actual outcomes: {actual_counts}")
        print(f"Model's most frequent predictions: {predicted_counts}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
