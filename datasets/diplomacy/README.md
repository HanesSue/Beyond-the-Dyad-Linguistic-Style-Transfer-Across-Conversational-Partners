# Diplomacy corpus

Beyond the Dyad uses the public Diplomacy messages collected by Peskov et al. (2020). The three JSONL splits are byte-for-byte copies of the [authors' release](https://github.com/DenisPeskoff/2020_acl_diplomacy/tree/c28c0870a639f6f388c87d4f8eaf685e64bfb1a6/data). Message counts sum the lengths of the `messages` arrays.

| File | Games | Messages | SHA-256 |
| --- | ---: | ---: | --- |
| [train.jsonl](train.jsonl) | 9 | 13,132 | `833d8ef26561bbd1921e9f5e730b2ebea0a59cec24f7082bfdcacb8b6cf30792` |
| [validation.jsonl](validation.jsonl) | 1 | 1,416 | `6b2ca19991c7596693ea61b34b1be2d6ada3fcf6710472edd4afed841556e571` |
| [test.jsonl](test.jsonl) | 2 | 2,741 | `2061ee7f040d6d8016760cb146d7821fef57cfbb4ced868f233e63854257c050` |
| **Total** | **12** | **17,289** | |

Each JSONL row contains a dialogue between two players within a game. To build a message-level history, expand the parallel `messages`, `speakers`, and `receivers` arrays, then sort by `absolute_message_index` within `game_id`. Identify players by game and player name together. The absolute index records message order. `years` and `seasons` describe game phases. The original sender and receiver labels describe intended and perceived truthfulness. [Upstream data dictionary](https://github.com/DenisPeskoff/2020_acl_diplomacy/blob/c28c0870a639f6f388c87d4f8eaf685e64bfb1a6/data/README.md) · [ConvoKit documentation](https://convokit.cornell.edu/documentation/diplomacy.html)

## Attribution and license

Denis Peskov, Benny Cheng, Ahmed Elgohary, Joe Barrow, Cristian Danescu-Niculescu-Mizil, and Jordan Boyd-Graber (2020). *It Takes Two to Lie: One to Lie, and One to Listen.* Proceedings of ACL 2020. [Paper and citation](https://aclanthology.org/2020.acl-main.353/)

The upstream corpus is distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The [upstream license text](LICENSE) accompanies the copied files.
