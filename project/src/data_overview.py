"""Produce reproducible descriptive statistics and plots for the local corpus.

Run from the repository root with Python, pandas, numpy, and matplotlib:
    python project/src/data_overview.py
"""

from bisect import bisect_right
from collections import Counter
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from describe_corpus import COUNTRY, ORDER
from diplomacy import DATA_DIR, LIST_FIELDS, ROOT, SPLITS, load_dialogues, load_messages
from markers import LABELS, marker_matrix


ASSETS = ROOT / "project" / "docs" / "assets" / "data-overview"
COUNTRIES = ("austria", "england", "france", "germany", "italy", "russia", "turkey")


def distribution(values):
    series = pd.Series(values, dtype=float).dropna()
    quantiles = series.quantile([0, .05, .25, .5, .75, .95, 1])
    return {"n": len(series), "mean": float(series.mean()),
            "quantiles": {str(q): float(value) for q, value in quantiles.items()}}


def exposure_history(messages):
    """All predictors use strictly earlier messages within the same game."""
    rows = []
    for game_id, game in messages.groupby("game_id", sort=True):
        events = list(game[["abs_index", "speaker", "receiver"]].itertuples(index=False, name=None))
        for player in COUNTRIES:
            inbox = {source: [] for source in COUNTRIES if source != player}
            last_outgoing = {}
            previous_partner = None
            n_outgoing = 0
            latest_unseen = None
            for index, sender, recipient in events:
                if recipient == player:
                    inbox[sender].append(index)
                elif sender == player:
                    same = inbox[recipient]
                    others = [times for source, times in inbox.items() if source != recipient]
                    latest_other = max((times[-1] for times in others if times), default=None)
                    previous_to_recipient = last_outgoing.get(recipient)
                    intervening_other = None
                    if previous_to_recipient is not None:
                        intervening_other = sum(len(times) - bisect_right(times, previous_to_recipient)
                                                for times in others)
                    latest_input = max((times[-1] for times in inbox.values() if times), default=None)
                    rows.append({
                        "game_id": int(game_id), "index": index, "player": player,
                        "recipient": recipient, "n_prior_outgoing": n_outgoing,
                        "n_sources": sum(bool(times) for times in inbox.values()),
                        "first_to_recipient": previous_to_recipient is None,
                        "switched": previous_partner != recipient if previous_partner is not None else None,
                        "last_incoming_from_recipient": bool(same and same[-1] == latest_input)
                        if latest_input is not None else None,
                        "same_lag": index - same[-1] if same else None,
                        "other_lag": index - latest_other if latest_other is not None else None,
                        "unseen_lag": index - latest_unseen if latest_unseen is not None else None,
                        "n_other_since_last_to_recipient": intervening_other,
                        "n_same_inputs": len(same),
                        "n_other_inputs": sum(map(len, others)),
                    })
                    last_outgoing[recipient] = index
                    previous_partner = recipient
                    n_outgoing += 1
                else:
                    latest_unseen = index
    return pd.DataFrame(rows)


def build_summary(dialogues, messages, exposure, markers):
    files, array_types, checks = [], {}, []
    all_roles = set()
    for split in SPLITS:
        path = DATA_DIR / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            all_roles.update((row["game_id"], name) for name in row["players"])
            for field in LIST_FIELDS:
                counts = array_types.setdefault(field, Counter())
                counts.update(type(value).__name__ for value in row[field])
            checks.append({
                "relative_contiguous": row["relative_message_index"] == list(range(len(row["messages"]))),
                "absolute_sorted": row["absolute_message_index"] == sorted(row["absolute_message_index"]),
                "players_consistent": all(name in row["players"] for name in row["speakers"] + row["receivers"]),
            })
        files.append({"split": split, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "rows": len(rows), "games": sorted({row["game_id"] for row in rows}),
                      "messages": sum(len(row["messages"]) for row in rows),
                      "empty_rows": sum(not row["messages"] for row in rows)})

    game_table = []
    for game_id, game in messages.groupby("game_id", sort=True):
        pairs = dialogues[dialogues.game_id == game_id]
        game_table.append({"game_id": int(game_id), "split": game.split.iloc[0],
                           "messages": len(game), "speakers": game.speaker.nunique(),
                           "nonempty_pairs": int((pairs.n_messages > 0).sum()),
                           "year_start": int(game.years.min()), "year_end": int(game.years.max()),
                           "median_words": float(game.n_tokens.median())})

    player_index = pd.MultiIndex.from_tuples(sorted(all_roles), names=["game_id", "speaker"])
    sent = messages.groupby(["game_id", "speaker"]).size().reindex(player_index, fill_value=0)
    received = messages.groupby(["game_id", "receiver"]).size().reindex(player_index, fill_value=0)
    partner_counts = messages.groupby(["game_id", "speaker"]).receiver.nunique().reindex(player_index, fill_value=0)
    known = messages[messages.receiver_labels.map(lambda value: type(value) is bool)]
    confusion = []
    for sender_value in (True, False):
        for receiver_value in (True, False, "NOANNOTATION"):
            confusion.append({"sender_truthful": sender_value, "receiver_label": receiver_value,
                              "count": int(((messages.sender_labels == sender_value)
                                            & (messages.receiver_labels == receiver_value)).sum())})
    score_by_phase = messages.groupby(["game_id", "speaker", "years", "seasons"]).score
    receiver_scores = score_by_phase.first().rename("receiver_score")
    compared = messages.merge(receiver_scores, left_on=["game_id", "receiver", "years", "seasons"],
                               right_index=True)
    abs_checks = [game.abs_index.tolist() == list(range(len(game)))
                  for _, game in messages.groupby("game_id")]
    later = exposure[~exposure.first_to_recipient]
    both = exposure[exposure.same_lag.notna() & exposure.other_lag.notna()]
    broad = sent[sent >= 40].index
    rates = []
    for name in markers:
        player_rates = markers[name].groupby([messages.game_id, messages.speaker]).mean().loc[broad]
        rates.append({"marker": name, "label": LABELS[name], "count": int(markers[name].sum()),
                      "rate": float(markers[name].mean()), "players_ge40": len(broad),
                      "player_p10": float(player_rates.quantile(.1)),
                      "player_median": float(player_rates.median()),
                      "player_p90": float(player_rates.quantile(.9))})

    split_profiles = []
    for split, part in messages.groupby("split", sort=False):
        split_profiles.append({"split": split, "messages": len(part),
                               "median_words": float(part.n_tokens.median()),
                               "sender_lie_rate": float(part.sender_labels.eq(False).mean()),
                               "receiver_missing_rate": float(part.receiver_labels.eq("NOANNOTATION").mean()),
                               "marker_rates": {name: float(markers.loc[part.index, name].mean())
                                                for name in ("contraction", "hedge", "question_mark")}})
    example_columns = ["game_id", "abs_index", "speaker", "receiver", "text", "years", "seasons",
                       "sender_labels", "receiver_labels"]
    example_rows = messages[(messages.game_id == 1) & messages.abs_index.isin([168, 268, 274, 275, 279, 281])]

    return {
        "date": "2026-11-09", "files": files, "games": game_table, "split_profiles": split_profiles,
        "examples": json.loads(example_rows[example_columns].to_json(orient="records", force_ascii=False)),
        "types": {name: dict(counter) for name, counter in array_types.items()},
        "integrity": {
            "aligned_arrays": bool(dialogues.arrays_aligned.all()),
            "dialogue_relative_indices_contiguous": all(c["relative_contiguous"] for c in checks),
            "dialogue_absolute_indices_sorted": all(c["absolute_sorted"] for c in checks),
            "dialogue_players_consistent": all(c["players_consistent"] for c in checks),
            "game_absolute_indices_contiguous": all(abs_checks),
            "duplicate_message_ids": int(messages.duplicated(["game_id", "abs_index"]).sum()),
            "duplicate_pair_rows": int(dialogues.assign(pair=dialogues.players.map(sorted).map(tuple))
                                       .duplicated(["game_id", "pair"]).sum()),
            "self_messages": int((messages.speaker == messages.receiver).sum()),
            "blank_messages": int(messages.text.str.strip().eq("").sum()),
            "null_cells": int(messages.isna().sum().sum()),
            "repeated_text_same_directed_pair": int(messages.duplicated(["game_id", "speaker", "receiver", "text"]).sum()),
            "score_phase_constant": bool(score_by_phase.nunique().eq(1).all()),
            "score_delta_checked": len(compared),
            "score_delta_mismatches": int((compared.score_delta != compared.score - compared.receiver_score).sum()),
        },
        "players": {"roles": len(all_roles), "active_speakers": len(sent[sent > 0]),
                    "sent_all": distribution(sent), "sent_active": distribution(sent[sent > 0]),
                    "received": distribution(received),
                    "receiver_count_histogram": {str(k): int(v) for k, v in partner_counts.value_counts().sort_index().items()},
                    "all_role_rows": [{"game_id": int(game), "country": country,
                                       "sent": int(sent.loc[(game, country)]),
                                       "received": int(received.loc[(game, country)])}
                                      for game, country in player_index]},
        "text": {"words": distribution(messages.n_tokens), "characters": distribution(messages.text.str.len()),
                 "total_whitespace_tokens": int(messages.n_tokens.sum()),
                 "messages_le7_words": int(messages.n_tokens.le(7).sum()),
                 "messages_le20_words": int(messages.n_tokens.le(20).sum()),
                 "country_mentions": int(messages.text.str.contains(COUNTRY, regex=True).sum()),
                 "order_notation": int(messages.text.str.contains(ORDER, regex=True).sum())},
        "pairs": {"all": distribution(dialogues.n_messages),
                  "nonempty": distribution(dialogues.loc[dialogues.n_messages > 0, "n_messages"]),
                  "directed_nonempty": messages.groupby(["game_id", "speaker", "receiver"]).ngroups},
        "labels": {"sender": {str(k): int(v) for k, v in messages.sender_labels.value_counts().items()},
                   "receiver": {str(k): int(v) for k, v in messages.receiver_labels.value_counts().items()},
                   "matrix": confusion, "receiver_annotated": len(known),
                   "agreement_annotated": int((known.sender_labels == known.receiver_labels).sum())},
        "phases": {"years": {str(k): int(v) for k, v in messages.years.value_counts().sort_index().items()},
                   "seasons": {str(k): int(v) for k, v in messages.seasons.value_counts().items()},
                   "score_range": [int(messages.score.min()), int(messages.score.max())],
                   "score_delta_range": [int(messages.score_delta.min()), int(messages.score_delta.max())]},
        "exposure": {"n": len(exposure), "first_to_recipient": int(exposure.first_to_recipient.sum()),
                     "switch_eligible": int(exposure.switched.notna().sum()),
                     "switch_count": int(exposure.switched.fillna(False).sum()),
                     "incoming_available": int(exposure.last_incoming_from_recipient.notna().sum()),
                     "last_incoming_from_recipient": int(exposure.last_incoming_from_recipient.fillna(False).sum()),
                     "heard_two_sources": int(exposure.n_sources.ge(2).sum()),
                     "same_and_other_available": len(both),
                     "unseen_control_available": int(exposure.unseen_lag.notna().sum()),
                     "warmup_coverage": [{"prior_outgoing_min": minimum,
                                           "outputs": int(((exposure.n_prior_outgoing >= minimum)
                                                           & exposure.same_lag.notna()
                                                           & exposure.other_lag.notna()).sum())}
                                          for minimum in (0, 3, 5, 10, 20)],
                     "later_to_recipient": len(later),
                     "other_since_previous_count": int(later.n_other_since_last_to_recipient.gt(0).sum()),
                     "other_since_previous": distribution(later.n_other_since_last_to_recipient),
                     "same_lag_both": distribution(both.same_lag), "other_lag_both": distribution(both.other_lag)},
        "markers": sorted(rates, key=lambda row: row["rate"], reverse=True),
    }


def draw_figures(summary, messages, exposure):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    palette = {"train": "#315a83", "validation": "#d08b30", "test": "#3c8d80"}
    table = summary["games"]
    axes[0, 0].bar([str(g["game_id"]) for g in table], [g["messages"] for g in table],
                   color=[palette[g["split"]] for g in table])
    axes[0, 0].set(title="A  Messages per game", xlabel="Game ID", ylabel="Messages")
    for split, color in palette.items():
        axes[0, 0].plot([], [], color=color, linewidth=6, label=split)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].hist(messages.n_tokens.clip(upper=100), bins=np.arange(0, 106, 5), color="#315a83", edgecolor="white")
    axes[0, 1].axvline(messages.n_tokens.median(), color="#d08b30", linewidth=2, label="Median 14")
    axes[0, 1].set(title="B  Message length", xlabel="Whitespace tokens (100+ pooled)", ylabel="Messages")
    axes[0, 1].legend(frameon=False)
    sorted_counts = sorted((row["sent"] for row in summary["players"]["all_role_rows"]), reverse=True)
    axes[1, 0].bar(np.arange(1, len(sorted_counts) + 1), sorted_counts, color="#3c8d80")
    axes[1, 0].set(title="C  Unequal contributions", xlabel="Player-game rank (84 roles)", ylabel="Messages sent")
    matrix = np.array([row["count"] for row in summary["labels"]["matrix"]]).reshape(2, 3)
    axes[1, 1].imshow(np.log1p(matrix), cmap="Blues", aspect="auto")
    for (i, j), value in np.ndenumerate(matrix):
        axes[1, 1].text(j, i, f"{value:,}", ha="center", va="center", color="white" if value > 10000 else "#142c46")
    axes[1, 1].set(xticks=range(3), xticklabels=["Truth", "Lie", "Missing"],
                   yticks=range(2), yticklabels=["Truth", "Lie"],
                   xlabel="Receiver perception", ylabel="Sender intention", title="D  Truthfulness annotations (counts)")
    fig.suptitle("Diplomacy corpus profile | 17,289 messages, 12 games", fontsize=16)
    fig.savefig(ASSETS / "corpus-profile.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    friendly = {"contraction": "Contractions / possessive 's", "question_mark": "Question mark",
                "we_pronoun": "we / us / our / let's", "hedge": "Hedge candidates", "exclamation_mark": "Exclamation mark",
                "emoji": "Emoji / emoticons", "ellipsis": "Ellipsis", "laughter": "Laughter",
                "all_lowercase": "All lowercase", "modal_request": "Modal + you", "thanks": "Thanks", "please": "Please"}
    rows = [row for row in summary["markers"] if row["marker"] != "curly_apostrophe"]
    axes[0].barh([friendly[row["marker"]] for row in rows], [100 * row["rate"] for row in rows], color="#315a83")
    axes[0].invert_yaxis()
    axes[0].set(title="A  Surface marker coverage", xlabel="Messages containing marker (%)")
    for i, row in enumerate(rows):
        axes[0].text(100 * row["rate"] + .6, i, f"{100 * row['rate']:.1f}%", va="center", fontsize=9)
    axes[0].set_xlim(0, 58)
    eligible = exposure[exposure.same_lag.notna() & exposure.other_lag.notna()]
    for column, color, label in (("same_lag", "#315a83", "Current recipient's latest input"),
                                 ("other_lag", "#d08b30", "Other partner's latest input")):
        x = np.sort(eligible[column].to_numpy())
        axes[1].step(x, np.arange(1, len(x) + 1) / len(x), color=color, label=label, linewidth=2)
    axes[1].set(xscale="log", xlim=(1, max(eligible.other_lag.max(), eligible.same_lag.max())),
                xlabel="Current index minus input index (log scale)", ylabel="Cumulative proportion",
                title=f"B  Input lags on the same {len(eligible):,} outputs")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Style measurement and cross-partner exposure", fontsize=16)
    fig.savefig(ASSETS / "style-and-exposure.png", dpi=180)
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    dialogues = load_dialogues()
    if not dialogues.arrays_aligned.all():
        raise ValueError("Parallel arrays have inconsistent lengths")
    messages = load_messages()
    exposure = exposure_history(messages)
    markers = marker_matrix(messages.text)
    summary = build_summary(dialogues, messages, exposure, markers)
    (ASSETS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    draw_figures(summary, messages, exposure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
