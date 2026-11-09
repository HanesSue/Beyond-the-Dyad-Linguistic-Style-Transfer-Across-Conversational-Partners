# 消息级 CSV 数据

每条消息占一条 CSV 记录，按 `game_id`、`amindex` 升序排列。本目录由仓库原始 JSONL 展开生成。

## 可行性及身份假设

将一个游戏中的国家角色作为固定玩家身份，可以直接利用消息里的国家名重建个人历史。推荐以 `(game_id, country)` 识别玩家。原始数据提供国家角色，真人身份、代打或游戏中途换人需要额外记录才能核实。本次采用每局玩家与国家一一对应的工作假设。

`amindex` 取自原始 `absolute_message_index`。12 场游戏各自从 0 连续编号，17,289 条消息的 `(game_id, amindex)` 全部唯一。因此它可以充当游戏内的顺序时间戳，比较先后及消息序号差。现实时间间隔需要真实时间戳。

主键为 **`(game_id, amindex)` 联合主键**。两列分别保留。直接相乘会发生碰撞，例如 `1 × 6 = 2 × 3`。跨游戏的消息先后也不能从 `amindex` 判断。

## 文件

| 文件 | 记录数，不含表头 | 游戏 |
| --- | ---: | --- |
| [messages.csv](messages.csv) | 17,289 | 全部 12 局 |
| [train.csv](train.csv) | 13,132 | 1、2、3、5、6、7、8、9、10 |
| [validation.csv](validation.csv) | 1,416 | 11 |
| [test.csv](test.csv) | 2,741 | 4、12 |

合并文件中的 `split` 对应原始官方切分。三份分割 CSV 的记录合起来与合并文件一致。

## 列定义

| 列 | 读取后的建议类型 | 定义 |
| --- | --- | --- |
| `game_id` | 整数 | 游戏 ID，与 `amindex` 联合唯一 |
| `amindex` | 整数 | 游戏内所有消息的全局序号，从 0 开始 |
| `sender` | 字符串 | 发信人的国家角色 |
| `receiver` | 字符串 | 收信人的国家角色 |
| `message` | 字符串 | 原始消息，保留原有换行、标点、空白与表情 |
| `relativeid` | 整数 | 同一 `(game_id, sender, receiver)` 方向下的消息序号，从 0 开始 |
| `dialogue_relativeid` | 整数 | 原始 `relative_message_index`，同一游戏同一对玩家的双向消息合并计数，从 0 开始 |
| `year` | 整数 | 游戏内年份，例如 1901 |
| `season` | 字符串 | `Spring`、`Fall` 或 `Winter` |
| `split` | 字符串 | `train`、`validation` 或 `test` |

`relativeid` 按“相同收发人”解释为同方向序号。`dialogue_relativeid` 保留原有的双向对话计数，便于与源文件对照。以下例子均来自游戏 1。

| `amindex` | `sender` | `receiver` | `relativeid` | `dialogue_relativeid` |
| ---: | --- | --- | ---: | ---: |
| 2 | austria | russia | 0 | 0 |
| 6 | russia | austria | 0 | 1 |
| 7 | austria | russia | 1 | 2 |

本导出表对应请求的消息、身份、顺序和游戏阶段字段，额外保存官方切分与原始双向序号。发送者和接收者的真诚标注、分数及分差继续保存在上级目录的 JSONL 中，按联合主键可关联。

## 编码与读取

文件采用 **UTF-8 with BOM**，方便 Excel 识别字符编码。字段使用标准 CSV 引号转义。消息内的真实换行保留在被引号包住的字段里，因此文件的物理行数可能大于消息记录数。使用 CSV 解析器读取即可还原记录。

使用 pandas 可以保留文本，并设置联合索引。

```python
import pandas as pd

messages = pd.read_csv(
    "datasets/diplomacy/csv/messages.csv",
    encoding="utf-8-sig",
    keep_default_na=False,
    dtype={
        "game_id": "int64",
        "amindex": "int64",
        "relativeid": "int64",
        "dialogue_relativeid": "int64",
        "year": "int64",
        "sender": "string",
        "receiver": "string",
        "message": "string",
        "season": "string",
        "split": "string",
    },
)
messages = messages.set_index(["game_id", "amindex"], verify_integrity=True)
```

`keep_default_na=False` 可以使合法的消息文本保持字符串，包括某条完整消息恰好为 `NA` 等特殊文本的情况。

## 重现转换

在仓库根目录运行

```bash
python project/src/export_messages_csv.py
```

该脚本复用现有的 `diplomacy.py` 加载器，先检查平行数组、联合主键、全局顺序和原始双向相对序号，再生成四份 CSV。每个文件发布前都逐字段回读比较，验证原文、数字字段和消息顺序。

数据来源和 CC BY 4.0 许可见 [上级目录说明](../README.md)。
