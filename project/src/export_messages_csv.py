"""Flatten the source dialogues into chronological, message-level CSV files.

Run from the repository root:
    python project/src/export_messages_csv.py
"""

import csv
import os
from pathlib import Path
import tempfile

from diplomacy import DATA_DIR, SPLITS, load_dialogues, load_messages


OUTPUT_DIR = DATA_DIR / "csv"
COLUMNS = (
    "game_id", "amindex", "sender", "receiver", "message", "relativeid",
    "dialogue_relativeid", "year", "season", "split",
)


def prepare_messages():
    dialogues = load_dialogues()
    if not dialogues.arrays_aligned.all():
        raise ValueError("The source dialogue arrays have different lengths")

    messages = load_messages()
    if messages.duplicated(["game_id", "abs_index"]).any():
        raise ValueError("Duplicate (game_id, absolute_message_index) keys")
    for game_id, game in messages.groupby("game_id", sort=False):
        if game.abs_index.tolist() != list(range(len(game))):
            raise ValueError(f"Non-contiguous message order in game {game_id}")
        if game.split.nunique() != 1:
            raise ValueError(f"Game {game_id} spans multiple dataset splits")

    original_order = messages.groupby(["game_id", "pair"]).cumcount()
    if not messages.relative_message_index.eq(original_order).all():
        raise ValueError("Original dialogue indices disagree with chronological order")

    messages["relativeid"] = messages.groupby(["game_id", "speaker", "receiver"]).cumcount()
    messages = messages.rename(columns={
        "abs_index": "amindex", "speaker": "sender", "text": "message",
        "relative_message_index": "dialogue_relativeid", "years": "year", "seasons": "season",
    })
    messages["year"] = messages.year.astype(int)
    return messages.loc[:, list(COLUMNS)]


def export_csv(table, output_path):
    """Use RFC-style quoting and verify every field before publishing a file."""
    expected = [tuple(str(value) for value in row)
                for row in table.itertuples(index=False, name=None)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                     dir=output_path.parent, prefix=".export-", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.writer(stream, dialect="excel")
            writer.writerow(COLUMNS)
            writer.writerows(table.itertuples(index=False, name=None))
            stream.flush()
            with temporary.open(encoding="utf-8-sig", newline="") as check:
                reader = csv.reader(check)
                if next(reader) != list(COLUMNS):
                    raise ValueError(f"CSV header changed in {output_path.name}")
                actual = [tuple(row) for row in reader]
            if actual != expected:
                raise ValueError(f"CSV round-trip changed data in {output_path.name}")
            stream.close()
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"{output_path.relative_to(DATA_DIR)}: {len(table):,} messages, round-trip verified")


def main():
    messages = prepare_messages()
    export_csv(messages, OUTPUT_DIR / "messages.csv")
    for split in SPLITS:
        export_csv(messages.loc[messages.split == split], OUTPUT_DIR / f"{split}.csv")


if __name__ == "__main__":
    main()
