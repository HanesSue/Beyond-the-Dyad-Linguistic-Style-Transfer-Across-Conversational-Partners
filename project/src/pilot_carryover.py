"""Rough pilot: does a style marker in P's message to B follow input from B, or from P's other partners?

For each message P -> B and each binary marker, a logistic GLM predicts whether the
message contains the marker from:

  B_in       marker in B's most recent message to P                  (same partner)
  O_in       marker in the most recent message to P from anyone != B (other partner)
  U_unseen   marker in the most recent message between two other players,
             which P never saw (placebo for shared game phase and topic)
  controls   logit of P's smoothed earlier rate of the marker, P's previous outgoing
             message, P's own last message to B / to the other partner before their
             reply (P's style echoed back), and log message length

Standard errors are clustered by player-game. This is a first look with rough regex
markers and single-message exposure, not a test of the project's hypotheses.

Run from the repository root:
    uv run python project/src/pilot_carryover.py
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from diplomacy import load_messages
from markers import LABELS, marker_matrix

ALPHA = 10.0        # pseudo-count shrinking P's early rate toward the corpus rate
MIN_PRIOR_OUT = 3   # P must have sent at least this many messages before
INBOX_LOOKBACK = 50
UNSEEN_LOOKBACK = 200

PREDICTORS = ["B_in", "O_in", "U_unseen", "self_prev", "echo_B", "echo_O", "logit_base", "loglen"]


def _last_before(indices, t):
    """Last element of the sorted list `indices` that is < t, or None."""
    p = np.searchsorted(indices, t) - 1
    return indices[p] if p >= 0 else None


def build_features(df: pd.DataFrame, X: np.ndarray):
    """Return target row ids, per-predictor (n, K) arrays, player ids and recency gaps."""
    K = X.shape[1]
    corpus_rate = X.mean(axis=0)
    zeros = np.zeros(K)
    out = {k: [] for k in ["i", "player", "base", "self_prev", "B_in", "O_in", "U_unseen",
                           "echo_B", "echo_O", "gap_B", "gap_O"]}
    for gid, g in df.groupby("game_id"):
        rows = g.index.to_numpy()
        S, R = g.speaker.to_numpy(), g.receiver.to_numpy()
        players = set(S) | set(R)
        count = {p: np.zeros(K) for p in players}
        n_out = dict.fromkeys(players, 0)
        prev_out = dict.fromkeys(players)
        sent_to = {}                          # (P, X) -> rows of P's messages to X
        last_from = {}                        # (P, X) -> row of X's last message to P
        inbox = {p: [] for p in players}      # P -> [(row, sender)]
        for j, (i, P, B) in enumerate(zip(rows, S, R)):
            other = next(((k, snd) for k, snd in reversed(inbox[P][-INBOX_LOOKBACK:]) if snd != B), None)
            unseen = next((rows[jj] for jj in range(j - 1, max(-1, j - UNSEEN_LOOKBACK), -1)
                           if S[jj] != P and R[jj] != P), None)
            b = last_from.get((P, B))
            if (n_out[P] >= MIN_PRIOR_OUT and prev_out[P] is not None
                    and b is not None and other is not None and unseen is not None):
                eb = _last_before(sent_to.get((P, B), []), b)
                eo = _last_before(sent_to.get((P, other[1]), []), other[0])
                out["i"].append(i)
                out["player"].append(f"{gid}:{P}")
                out["base"].append((count[P] + ALPHA * corpus_rate) / (n_out[P] + ALPHA))
                out["self_prev"].append(X[prev_out[P]])
                out["B_in"].append(X[b])
                out["O_in"].append(X[other[0]])
                out["U_unseen"].append(X[unseen])
                out["echo_B"].append(X[eb] if eb is not None else zeros)
                out["echo_O"].append(X[eo] if eo is not None else zeros)
                # load_messages() sorts by game and order, so row ids are consecutive within a game
                out["gap_B"].append(i - b)
                out["gap_O"].append(i - other[0])
            count[P] += X[i]
            n_out[P] += 1
            prev_out[P] = i
            sent_to.setdefault((P, B), []).append(i)
            last_from[(B, P)] = i
            inbox[B].append((i, P))
    return {k: np.asarray(v) for k, v in out.items()}


def fit_marker(F, X, loglen, k):
    y = X[F["i"], k]
    base = F["base"][:, k]
    d = pd.DataFrame({
        "B_in": F["B_in"][:, k], "O_in": F["O_in"][:, k], "U_unseen": F["U_unseen"][:, k],
        "self_prev": F["self_prev"][:, k], "echo_B": F["echo_B"][:, k], "echo_O": F["echo_O"][:, k],
        "logit_base": np.log(base / (1 - base)), "loglen": loglen[F["i"]],
    })[PREDICTORS]
    groups = pd.factorize(F["player"])[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.GLM(y, sm.add_constant(d), family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
    # Rare markers can be perfectly predicted by a predictor; the estimate then runs off to ±inf.
    separated = any(issubclass(w.category, PerfectSeparationWarning) for w in caught) \
        or (fit.params.abs() > 10).any()
    return y.mean(), fit, separated


def main():
    df = load_messages()
    M = marker_matrix(df.text)
    X = M.to_numpy().astype(float)
    loglen = np.log1p(df.n_tokens.to_numpy())
    F = build_features(df, X)

    print(f"analysable P->B messages: {len(F['i'])} of {len(df)}")
    print("median distance (messages, game-wide order) to B's last input:", np.median(F["gap_B"]),
          "| to the other partner's last input:", np.median(F["gap_O"]))

    rows = []
    for k, name in enumerate(M.columns):
        rate, fit, separated = fit_marker(F, X, loglen, k)
        ci = fit.conf_int()
        rec = {"marker": name, "label": LABELS[name], "rate": round(rate, 3)}
        for v in ["B_in", "O_in", "U_unseen", "self_prev"]:
            rec[v] = f"{fit.params[v]:+.2f} [{ci.loc[v, 0]:+.2f}, {ci.loc[v, 1]:+.2f}]"
        rec["p(O_in=U_unseen)"] = f"{float(np.asarray(fit.t_test('O_in - U_unseen = 0').pvalue)):.3f}"
        rec["note"] = "separation: not estimable" if separated else ""
        rows.append(rec)

    print("\nlog-odds coefficients with 95% CI (clustered by player-game)")
    print(pd.DataFrame(rows).set_index("marker").to_string())


if __name__ == "__main__":
    main()
