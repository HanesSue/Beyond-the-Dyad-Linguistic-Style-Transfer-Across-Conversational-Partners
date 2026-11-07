"""Descriptive checks of the Diplomacy corpus.

Run from the repository root:
    uv run python project/src/describe_corpus.py
"""
import numpy as np
import pandas as pd

from diplomacy import load_dialogues, load_messages
from markers import LABELS, marker_matrix

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

COUNTRY = (r"(?i)\b(?:england|france|germany|italy|austria|russia|turkey|english|french|"
           r"german|italian|austrian|russian|turkish|eng|fra|ger|ita|aus|aut|rus|tur)\b")
ORDER = r"(?i)\b[af]\s+[a-z]{3}\s*(?:-|->|s|h|c)\s*"


def section(title):
    print(f"\n== {title} ==")


def integrity(dlg, df):
    section("size and integrity")
    print(dlg.groupby("split").agg(dialogues=("n_messages", "size"),
                                   messages=("n_messages", "sum"),
                                   games=("game_id", "nunique")))
    print("parallel arrays aligned in every row:", dlg.arrays_aligned.all())
    print("dialogue rows with no messages:", (dlg.n_messages == 0).sum())
    unordered = dlg.assign(pair=dlg.players.map(lambda p: tuple(sorted(p))))
    print("player pairs per game:", dlg.groupby("game_id").players.nunique().unique().tolist(),
          "| duplicated (game, pair) rows:", unordered.duplicated(["game_id", "pair"]).sum())
    g = df.groupby("game_id")
    complete = (g.abs_index.min() == 0) & (g.abs_index.max() == g.size() - 1) & (g.abs_index.nunique() == g.size())
    print("abs_index is a complete 0..N-1 order in every game:", complete.all())
    print("speaker == receiver:", (df.speaker == df.receiver).sum())
    for gid, gg in g:
        silent = set(gg.receiver) - set(gg.speaker)
        if silent:
            print(f"game {gid}: players who only receive: {sorted(silent)}")
    dup = df[df.duplicated(["game_id", "speaker", "receiver", "text"], keep=False)]
    print("repeated (game, speaker, receiver, text):",
          df.duplicated(["game_id", "speaker", "receiver", "text"]).sum(),
          "| most common:", dup.text.value_counts().head(6).to_dict())


def fields(df):
    section("field semantics")
    k = df.groupby(["game_id", "speaker", "years", "seasons"]).score.nunique()
    print("score constant within (game, speaker, year, season):", (k == 1).all())
    rec = df.groupby(["game_id", "speaker", "years", "seasons"]).score.first().rename("receiver_score")
    m = df.merge(rec, left_on=["game_id", "receiver", "years", "seasons"], right_index=True)
    print(f"score_delta == speaker score - receiver score: {(m.score_delta == m.score - m.receiver_score).mean():.3f}"
          f" (n={len(m)})")
    print("sender_labels:", df.sender_labels.value_counts().to_dict(),
          f"| lie rate {(df.sender_labels == False).mean():.3f}")  # noqa: E712
    print("receiver_labels:", df.receiver_labels.value_counts().to_dict())
    print("years:", df.years.value_counts().sort_index().to_dict())
    print("seasons:", df.seasons.value_counts().to_dict())


def distributions(df):
    section("distributions")
    g = df.groupby("game_id")
    print(pd.DataFrame({"messages": g.size(), "speakers": g.speaker.nunique(),
                        "pairs": g.pair.nunique(),
                        "years": g.years.agg(lambda s: f"{min(s)}-{max(s)}")}).T)
    q = [.05, .25, .5, .75, .95]
    print("tokens per message:", df.n_tokens.quantile(q).to_dict())
    print("chars per message:", df.text.str.len().quantile(q).to_dict())
    print("messages per player-game:", df.groupby("player").size().describe().round(1).to_dict())
    print("distinct receivers per player-game:",
          df.groupby("player").receiver.nunique().value_counts().sort_index().to_dict())
    print("messages per (game, pair):",
          df.groupby(["game_id", "pair"]).size().describe().round(1).to_dict())


def exposure(df):
    """How interleaved are a player's conversations with different partners?"""
    section("exposure structure for P->B messages")
    recs = []
    for _, g in df.groupby("game_id"):
        S, R, A = g.speaker.to_numpy(), g.receiver.to_numpy(), g.abs_index.to_numpy()
        for P in set(S) | set(R):
            inbox, last_to, prev_partner = [], {}, None
            for a, s, r in zip(A, S, R):
                if r == P:
                    inbox.append((a, s))
                elif s == P:
                    since = last_to.get(r, -1)
                    recs.append(dict(
                        first_to_B=r not in last_to,
                        switch=prev_partner is not None and prev_partner != r,
                        last_in_from_B=inbox[-1][1] == r if inbox else np.nan,
                        from_others=sum(x > since and snd != r for x, snd in inbox),
                        heard_from=len({snd for _, snd in inbox}),
                    ))
                    last_to[r], prev_partner = a, r
    ex = pd.DataFrame(recs)
    later = ex[~ex.first_to_B]
    print("first message P ever sends to B:", round(ex.first_to_B.mean(), 3))
    print("consecutive outgoing messages switch partner:", round(ex.switch.mean(), 3))
    print("P's most recent incoming message came from B:", round(ex.last_in_from_B.mean(), 3))
    print("messages from others since P last wrote to B: share >=1 =",
          round((later.from_others > 0).mean(), 3),
          "| quantiles", later.from_others.quantile([.5, .75, .9]).to_dict())
    print("P had already heard from >=2 partners:", round((ex.heard_from >= 2).mean(), 3))


def content(df):
    section("game-specific content (to mask before measuring style)")
    print(f"mentions a country: {df.text.str.contains(COUNTRY, regex=True).mean():.3f}")
    print(f"contains order notation (e.g. 'A PAR - BUR'): {df.text.str.contains(ORDER, regex=True).mean():.3f}")


def marker_inventory(df):
    section("style marker rates; spread across player-games with >=40 messages")
    M = marker_matrix(df.text)
    n = df.groupby("player").size()
    big = n[n >= 40].index
    rows = []
    for name in M.columns:
        rates = M[name].groupby(df.player).mean().loc[big]
        rows.append(dict(marker=name, label=LABELS[name], overall=M[name].mean(),
                         p10=rates.quantile(.1), median=rates.median(), p90=rates.quantile(.9)))
    print(f"player-games included: {len(big)}")
    print(pd.DataFrame(rows).set_index("marker").round(3).to_string())


if __name__ == "__main__":
    dialogues, messages = load_dialogues(), load_messages()
    integrity(dialogues, messages)
    fields(messages)
    distributions(messages)
    exposure(messages)
    content(messages)
    marker_inventory(messages)
