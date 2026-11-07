"""Load the Diplomacy corpus (Peskov et al., 2020) as one row per message."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "diplomacy"
SPLITS = ("train", "validation", "test")

# Per-message fields stored as parallel arrays in each dialogue row.
LIST_FIELDS = (
    "messages", "sender_labels", "receiver_labels", "speakers", "receivers",
    "absolute_message_index", "relative_message_index", "seasons", "years",
    "game_score", "game_score_delta",
)


def load_dialogues(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per (game, player pair) dialogue, as stored in the JSONL files."""
    rows = []
    for split in SPLITS:
        with open(data_dir / f"{split}.jsonl", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                d = json.loads(line)
                lengths = {len(d[k]) for k in LIST_FIELDS}
                rows.append({
                    "split": split, "line": line_no, "game_id": d["game_id"],
                    "players": tuple(d["players"]), "n_messages": len(d["messages"]),
                    "arrays_aligned": len(lengths) == 1,
                })
    return pd.DataFrame(rows)


def load_messages(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per message, sorted by game and in-game order.

    Columns keep the upstream names, except:
      speaker / receiver    singular forms of `speakers` / `receivers`
      abs_index             int version of `absolute_message_index` (total order within a game)
      score, score_delta    ints: speaker's supply centres, speaker minus receiver
      pair                  unordered player pair, sorted
      player                "game_id:country", the only player identity the corpus offers
    """
    msgs = []
    for split in SPLITS:
        with open(data_dir / f"{split}.jsonl", encoding="utf-8") as f:
            for d in map(json.loads, f):
                for i in range(len(d["messages"])):
                    m = {k: d[k][i] for k in LIST_FIELDS}
                    m.update(split=split, game_id=d["game_id"])
                    msgs.append(m)
    df = pd.DataFrame(msgs).rename(columns={
        "messages": "text", "speakers": "speaker", "receivers": "receiver",
    })
    df["abs_index"] = df.absolute_message_index.astype(int)
    df["score"] = df.game_score.astype(int)
    df["score_delta"] = df.game_score_delta.astype(int)
    df["pair"] = [tuple(sorted(p)) for p in zip(df.speaker, df.receiver)]
    df["player"] = df.game_id.astype(str) + ":" + df.speaker
    df["n_tokens"] = df.text.str.split().str.len()
    return df.sort_values(["game_id", "abs_index"]).reset_index(drop=True)
