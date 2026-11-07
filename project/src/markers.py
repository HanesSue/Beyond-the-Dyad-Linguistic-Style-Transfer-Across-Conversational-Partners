"""Binary surface style markers: does a message contain the marker at least once?

These are rough regex detectors for a first look, not validated measures.
"""
import pandas as pd

EMOJI = r"[\U0001F300-\U0001FAFF☀-➿]|(?<!\w)[:;]-?[)(DPp](?!\w)"

MARKERS = {
    "modal_request": r"(?i)\b(?:could|would|can|will) you\b",
    "hedge": r"(?i)\b(?:maybe|perhaps|probably|possibly|might)\b|\bi (?:think|guess)\b",
    "we_pronoun": r"(?i)\b(?:we|us|our|ours|let['’]?s)\b",
    "please": r"(?i)\b(?:please|pls|plz)\b",
    "thanks": r"(?i)\b(?:thanks|thank you|thx|ty)\b",
    "question_mark": r"\?",
    "exclamation_mark": r"!",
    "ellipsis": r"\.\.\.|…",
    "emoji": EMOJI,
    "laughter": r"(?i)\b(?:lol|lmao|ha(?:ha)+h?|hehe)\b",
    "all_lowercase": None,  # has lowercase letters and no uppercase letters
    "contraction": r"(?i)\b\w+['’](?:s|t|re|ve|ll|d|m)\b",
    # Set by the player's keyboard (smart punctuation), so partners should not
    # change it. Used as a negative control, not as a style measure.
    "curly_apostrophe": r"’",
}

LABELS = {
    "modal_request": "情态请求 could/would/can/will you",
    "hedge": "模糊语 maybe/perhaps/probably/might/I think",
    "we_pronoun": "we/us/our/let's",
    "please": "please",
    "thanks": "thanks",
    "question_mark": "问号",
    "exclamation_mark": "感叹号",
    "ellipsis": "省略号",
    "emoji": "表情符号",
    "laughter": "笑声 lol/haha",
    "all_lowercase": "全小写",
    "contraction": "缩略形式 I'm/don't",
    "curly_apostrophe": "弯引号 ’（键盘，对照）",
}

NEGATIVE_CONTROLS = ("curly_apostrophe",)


def marker_matrix(texts: pd.Series) -> pd.DataFrame:
    """Boolean DataFrame, one column per marker, aligned with `texts`."""
    out = {}
    for name, pattern in MARKERS.items():
        if pattern is None:
            out[name] = texts.str.contains(r"[a-z]") & ~texts.str.contains(r"[A-Z]")
        else:
            out[name] = texts.str.contains(pattern, regex=True)
    return pd.DataFrame(out, index=texts.index)
