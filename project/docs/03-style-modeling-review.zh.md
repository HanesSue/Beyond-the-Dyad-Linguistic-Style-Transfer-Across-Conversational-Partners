# 自然语言风格建模文献综述

## 从语言特征、风格嵌入到可控生成与动态适应

**整理日期：2026 年 11 月 9 日**  
**用途：CCM 项目 Beyond the Dyad 的方法调研与建模参考**

## 摘要

自然语言风格建模研究的是表达方式的规律，包括词语选择、句法结构、礼貌与正式程度、标点习惯，以及随作者、伙伴和交流情境变化的表达倾向。相关研究分布在计算文体学、社会语言学、心理语言学、自然语言处理和认知建模中。各领域虽然都使用“风格”一词，建模目标却有明显差别。有的研究测量两段文本的相似性，有的识别作者，有的生成指定风格的文本，有的解释人为什么在接触他人之后改变表达。

本文按方法梳理这些研究，覆盖可解释特征、统计与层级模型、监督学习、神经风格嵌入、内容与风格解耦、文本改写、个性化生成、推理时控制，以及认知适应模型。文献以原始论文和正式发表的代表性成果为主，并补充已公开的近期研究。它是一篇选择性的、面向方法的综述，不以论文数量或某个基准上的排名替代方法比较。最后结合 Diplomacy 短消息，讨论哪些方法适合测量风格，哪些方法能够进一步检验跨伙伴迁移的认知假设。

## 阅读导航

- 第一至二节建立概念地图，先分清每类模型到底在解决什么问题。
- 第三至六节讨论如何把自然语言变成风格特征或向量，是本项目选择测量方法的主要依据。
- 第七至十节讨论如何生成、修改和控制语言风格，帮助理解语言组件可以承担什么工作。
- 第十一节讨论经验如何改变表达，把文本表征与认知模型连接起来。
- 第十二至十三节比较评估方法和研究缺口。
- 第十四节原为对本项目的具体判断，已并入[方法文档](02-method.zh.md)。

## 一、首先界定“风格”以及模型要解释的对象

### 1.1 风格包含多个层次

计算文体学通常从词频、字符序列、句法和篇章特征描述写作方式。Biber 的多维度分析进一步表明，口语与书面语之间的差别可以由多组共同变化的语言特征刻画，不能完全压缩为一条“正式—非正式”轴。不同维度可能对应参与性、信息密度等不同功能。[文献 2](https://doi.org/10.1017/CBO9780511621024)、[文献 3](https://doi.org/10.1002/asi.21001)

在具体研究中，至少需要区分以下四个对象。

| 对象 | 直观例子 | 适合问的问题 |
|---|---|---|
| **Register / situational style · 语域与情境风格** | 同一个人在论文中写得正式，在聊天中使用缩写 | 任务和场景如何改变表达？ |
| **Authorial style / idiolect · 作者风格与个人语言习惯** | 某人长期偏爱短句、特定标点或词语搭配 | 哪些习惯稳定地区分不同的人？ |
| **Local stylistic choice · 局部表达选择** | 此次请求采用疑问句还是祈使句 | 相同交流目的下，为什么选这种说法？ |
| **Adaptation / alignment · 适应与对齐** | 听过 A 的表达后，P 后续更常采用相似形式 | 表达如何随经历变化，影响是否迁移到 B？ |

这些对象可以相关，但不应直接互换。一个擅长识别作者的模型，可能主要保留长期稳定特征。一个用于改写礼貌程度的模型，可能只学到某个属性。它们是否能测量同一个人在几条消息之间的变化，需要另行验证。

### 1.2 风格与内容只能在明确任务下相对分开

以下是本文自拟的示例。

> A. “Could you support my move to Belgium?”  
> B. “Support my move to Belgium.”  
> C. “Could you support my move to Munich?”

A 与 B 请求的行动相近，表达的间接程度不同。A 与 C 使用同类句式，地点不同。一个用于测量请求方式的指标，应当感受到 A/B 的差异，同时尽量不被 Belgium/Munich 的变化主导。

但 A/B 也带来礼貌、权力关系和请求强度上的差异。“保持内容不变”需要说明究竟保持了命题、交流目的，还是实际交际效果。文本风格迁移文献中的 sentiment transfer 还可能改变评价本身，因此不能把所有情感变化都当作纯粹的形式变化。Jin 等的综述提供了该领域的总体方法分类，也揭示了不同任务对风格定义的依赖。[文献 1](https://doi.org/10.1162/coli_a_00426)

### 1.3 三种“风格向量”承担不同工作

本文用三个符号避免混淆。

$$
\mathbf z_t=e(u_t),\qquad
\mathbf s_{P,t}=\operatorname{Update}(H_{P,\lt t}),\qquad
u_{\mathrm{new}}\sim p_\psi(u\mid c,\mathbf v).
$$

- **Text representation · 文本表征** $\mathbf z_t$ 是从已经出现的消息 $u_t$ 提取的风格测量值。
- **Latent adaptive state · 潜在适应状态** $\mathbf s_{P,t}$ 是根据 P 的历史经历推断的当前状态，不直接等于某一条消息的向量。
- **Generation control · 生成控制量** $\mathbf v$ 用于让生成器在给定内容 $c$ 时采用某种表达方式。

三者可能通过模型连接。不能仅因它们都是向量，就把测量结果当成人脑中的真实状态，或把控制大模型的方向当作已经验证的心理维度。

## 二、主要方法的全景地图

下面的分类按功能组织，同一篇论文可能连接多个类别。

| 方法家族 | 主要输入 | 学习或计算方式 | 主要输出 | 最有价值的用途 | 主要限制 |
|---|---|---|---|---|---|
| 可解释语言特征与 stylometry | 文本、词表、词性或句法分析 | 计数、比例、规则、标准化 | 特征向量、距离 | 知道到底是哪种表达变化了 | 特征覆盖和短文本噪声 |
| 多维度与因子模型 | 多篇文本的特征矩阵 | 因子分析等 | 少数共同变化的维度 | 概括语域结构 | 因子含义需要解释，依赖语料 |
| 统计与层级模型 | 特征计数、作者、伙伴、情境 | 回归、部分汇聚、概率生成 | 效应、分布、预测概率 | 区分本人基线、伙伴与情境 | 需事先定义观测量和结构 |
| 风格属性分类 | 文本与人工或弱监督标签 | 线性分类器、神经编码器、多任务训练 | 正式、礼貌等属性得分 | 把特定属性量化 | 标签覆盖不等于全部风格 |
| 作者与对比式风格嵌入 | 文本及作者、会话或配对关系 | 度量学习、对比学习 | 稠密向量 | 开放式相似性与检索 | 可能混入内容和身份 |
| 可解释神经风格表征 | 文本与语言化的属性标注 | 属性预测、蒸馏 | 有描述的多维分数 | 兼顾覆盖和可解释性 | 标签来源与校准问题 |
| 潜变量解耦 | 平行或非平行的不同风格文本 | 自编码、VAE、对抗训练 | 内容码、风格码、改写文本 | 学习可组合的生成表示 | 分解通常不唯一 |
| 编辑、检索与释义生成 | 原句、目标语料或参考文本 | 删除标记、检索、改写 | 风格转换后的文本 | 不要求显式完整解耦 | 可能改变事实或删掉信息 |
| 条件语言模型与参数适配 | 上下文、风格标签或人物信息 | SFT、prefix、LoRA 等 | 下一个词的概率与文本 | 稳定地产生目标表达 | 训练目标未必对应人的机制 |
| 提示与个性化检索 | 指令、少量风格范例、用户历史 | 上下文条件化、历史检索 | 模仿风格的文本 | 原型实现和开放式风格 | 提示敏感，内容泄漏 |
| 解码与激活控制 | 基础生成器、属性模型或控制向量 | 调整 token 分布或隐藏激活 | 受控文本 | 无需为每种风格完整训练模型 | 控制强度与流畅性相互牵制 |
| 认知学习与动态模型 | 按时间排列的真实经历及后续行为 | 贝叶斯更新、误差学习、记忆检索 | 下一次表达的概率或分布 | 检验迁移和学习机制 | 需要能区分机制的数据 |

## 三、可解释特征：把风格表示为可检查的语言行为

### 3.1 从字符、功能词到句法与篇章

传统 stylometry 的常见特征包括字符或词语 n-gram、功能词比例、词汇丰富度、词长、句长、标点、词性序列和句法模式。字符 n-gram 是连续几个字符构成的片段，例如某些缩写或标点组合，它不要求完整句法分析，因此常用于作者识别。更复杂的特征可以描述从句、主被动、请求句式或篇章连接方式。[文献 3](https://doi.org/10.1002/asi.21001)

一个最小的风格向量可以写为

$$
\mathbf z(u)=\bigl[\text{平均句长},\ \text{问号比例},\ \text{第一人称复数比例},\ \text{缩写比例},\ \text{情态请求比例}\bigr].
$$

这些维度的分母不一定相同。功能词频率可以按总词数归一化。情态请求更适合在可识别的请求中计算。消息中没有请求时，不能直接把“没有使用情态请求”解释成偏爱直接命令。

**Linguistic Inquiry and Word Count（LIWC）** 是词典式分析的重要代表。它根据词语与类别的对应关系计算文本指标，类别涵盖功能词及心理、社会和主题相关内容。因此，应选择与研究问题对应的类别，不能将全部 LIWC 输出统称为纯风格。[文献 44](https://liwc.app/help/psychometrics-manuals)

Tokenizer 在这里负责切分文本，不会自行判定风格。切分之后，还需要规则、词典、句法分析器或训练好的编码器，才能得到可用于比较的表示。

特征向量的优势在于能追问变化发生在哪里。例如，“P 更像 A”可以分解为 P 的问句、缩写或代词使用发生了哪些变化。代价是研究者必须选择特征，规则也可能误识别语境。词表中的 hedge 候选词，并不保证在每个位置都承担缓和功能。

### 3.2 Burrows’s Delta：用标准化词频比较文本

Burrows 的 Delta 用常见词的相对频率描述文本，再把各维度标准化。经典形式可以表示为

$$
\Delta(x,y)=\frac{1}{d}\sum_{k=1}^{d}\lvert z_k(x)-z_k(y)\rvert.
$$

这里 $z_k$ 表示第 $k$ 个词频相对于参考语料的标准分数。标准化使极常见词不至于只因数值大而主导距离。原始研究用于文学作者归属，文本长度和体裁会影响可靠性。[文献 4](https://doi.org/10.1093/llc/17.3.267)

这类方法适合回答“两个文本的词频轮廓有多接近”。如果把它用于十几个词的聊天消息，许多维度只能取零或一次出现，距离会很不稳定。更长的窗口可以减小噪声，但也会平均掉短时变化。

### 3.3 Language Style Matching：比较功能词使用的相似程度

LSM 通常比较代词、冠词、介词等功能词类别的使用比例，再将各类别的匹配程度汇总。常见的归一化形式为

$$
\operatorname{LSM}_k
=1-\frac{\lvert p_{A,k}-p_{B,k}\rvert}{p_{A,k}+p_{B,k}+\epsilon}.
$$

两方比例越接近，得分越高。Ireland 与 Pennebaker 将这一路径用于作文、书信和诗歌中的语言同步。[文献 5](https://doi.org/10.1037/a0020386)

**相似性不包含变化方向。** 两个人一开始就相似，也会有高 LSM。在极短文本中，两边都没有某个类别也可能显得高度匹配。研究适应时还要比较暴露前后的变化、本人原有习惯和消息长度。

### 3.4 从匹配到有方向的 coordination

Danescu-Niculescu-Mizil 等研究语言协调与权力差异，使用类似以下的条件比较。

$$
\Pr(P\text{ 在回复中使用标记}\mid A\text{ 使用该标记})
-\Pr(P\text{ 在回复中使用标记}).
$$

这里比较的是 P 的回复是否随 A 的上一条表达而改变，包含了方向和本人基线。它比静态相似度更接近对齐问题，但仍可能受回复目的及共同情境影响。[文献 6](https://www.cs.cornell.edu/~cristian/Echoes_of_power_files/echoes_of_power.pdf)

### 3.5 内容遮盖与语法轮廓

POSNoise 保留选定的功能表达，把许多承载内容的词替换为词性标记，再交给下游方法。它的意义是减少模型通过主题词“认作者”的机会，而非保证主题信息全部消失。[文献 8](https://arxiv.org/abs/2005.06605)

例如，针对本项目可以尝试将地名替换为统一标记，让下面两句的地点差异减弱。

```text
Could you support Belgium?
Could you support Munich?

→ Could you VERB LOCATION?
```

这个例子是针对项目的简化示意，不是 POSNoise 原论文的完整规则。遮盖过多会同时删掉有用的风格线索，因此需要比较原文和遮盖后表征的表现。

## 四、统计模型：把“谁说的、对谁说、在什么情境说”放进模型

### 4.1 多维度分析：从许多特征中发现共同变化

Biber 的路线先测量一批可解释特征，再用因子分析识别共同变化的结构。例如，一些代词、缩略形式和互动表达可能共同出现。因子得分概括这种组合，不必预先把所有文本贴上“正式”或“随意”的标签。[文献 2](https://doi.org/10.1017/CBO9780511621024)

这很适合探索“哪些表达倾向一起变化”。不过，提取出的因子含义取决于特征、语域和样本。PCA 可以作为另外一种降维工具，但不能因此把它与原始多维度分析的理论解释完全等同。

### 4.2 稀疏加性生成模型：分开表示多个文本来源

SAGE 为每个因素学习相对于背景词分布的对数频率偏移，再把不同因素的偏移相加。它提供了一种可解释的概率文本建模方式，并用稀疏性减少过拟合。[文献 9](https://research.google/pubs/sparse-additive-generative-models-of-text/)

借用其加性思想，一个用于讲解的多因素表达是

$$
\log p(w\mid a,c)
=b_w+\eta_{a,w}+\xi_{c,w}-\log Z_{a,c}.
$$

$b_w$ 是词的背景倾向，$\eta_{a,w}$ 是作者相关偏移，$\xi_{c,w}$ 是主题或情境相关偏移，$Z$ 保证概率和为 1。这个作者—情境例子是说明性扩展。

它的启发是，同一个词的出现可以有多个来源。但给参数命名为“作者因素”并不自动保证它代表纯风格。若作者始终只谈一种主题，数据就难以区分两种解释。

### 4.3 WHAM：显式处理基线、长度和对齐

Doyle 与 Frank 的 Word-Based Hierarchical Alignment Model（WHAM）以消息中的标记次数和总词数为观测量，在词语、类别和对话参与者等层级估计基线与对齐。研究区分了复用同一个词和改用同类别其他词，也检查简短反馈语的影响。结果显示，抽象类别对齐不能直接由词语复用推出。[文献 7](https://aclanthology.org/P16-1050/)

对本项目来说，这种模型有两项价值。第一，某人本来就频繁使用问句，不会只因恰好回复了问句而被判断为强对齐。第二，长消息有更多机会出现标记，模型可以把这一点纳入观测过程。

**统计层级模型和认知层级模型也要区分。** 为参与者设置随机效应，是在统计上处理个体差异。若要解释人的学习，还需要规定经历怎样改变内部状态，以及内部状态怎样预测下一次表达。

## 五、监督学习：把特定风格属性变成预测任务

### 5.1 传统分类器与神经属性预测

这类方法输入一段文本，输出某种属性的类别或得分。早期研究常用词语、句法特征加线性模型，后来也可用 CNN、RNN 或 Transformer 编码器。模型通过带标签的数据学习“什么语言形式通常与这个属性一起出现”。

Danescu-Niculescu-Mizil 等的礼貌研究，将语言策略转化为可计算特征，用人工标注的请求检验礼貌识别及其社会关联。它说明礼貌可以被部分操作化，也说明这类判断依赖请求语境。[文献 10](https://aclanthology.org/P13-1025/)

输入 “Could you …?”，输出“礼貌”的概率，不等于预测这个人下一次为什么会用它。分类器承担测量工作。其输出是否经过概率校准、是否适用于游戏谈判中的反讽或威胁，需要另外检查。

### 5.2 多任务与多属性风格建模

XSLUE 汇集多种风格相关任务，研究共享学习如何帮助跨属性理解。它提醒我们，同一文本可以同时在多个维度上有特点，风格标签并非互斥类别。[文献 11](https://aclanthology.org/2021.acl-long.185/)

例如，一段消息可以同时正式、礼貌而负面。用多个属性分数构成向量，比只设“正式/非正式”更宽。但它仍受已标注属性的范围限制，也不能假设所有分数都是相互独立、可等权相加的心理量。

**选择条件。** 当问题明确针对礼貌、正式程度或某种句式时，属性预测具有较清晰的解释。若尚不知道哪个维度会发生变化，可把它作为向量的一部分，再与开放式嵌入比较。

## 六、神经风格嵌入：不预先枚举全部风格维度

### 6.1 对比学习的基本设计

编码器将文本变成向量。训练时让某些文本更接近，让另一些文本更远。一个示意性的 triplet loss 是

$$
\mathcal L
=\max\lbrace 0,\ d(e(u),e(u^+))-d(e(u),e(u^-))+m\rbrace.
$$

$u^+$ 是应当相似的样本，$u^-$ 是应当不同的样本，$m$ 是间隔。**真正决定模型学到什么的是配对依据。** 同作者、同风格和同主题，是三种不同的正例定义。

### 6.2 LUAR：学习跨文本的作者表征

LUAR 将同一作者的一组文本编码、聚合成一个向量，并用作者关系进行度量学习。架构包含文本编码、跨文本自注意力及聚合。其跨 Amazon、fanfiction 和 Reddit 的研究发现，作者表征可以跨域迁移，但不同迁移组合的效果有差别。[文献 12](https://aclanthology.org/2021.emnlp-main.70/)

**输入与输出。** 输入可以是一个作者的多篇文本，输出是用于作者验证的表征。聚合能减少单条消息的偶然性，也可能平滑掉我们关心的短时适应。因此，它适合作为作者习惯基线或候选表征，需要先验证局部变化的敏感度。

### 6.3 控制会话内容的风格嵌入

Wegmann、Schraagen 与 Nguyen 利用作者和会话关系构造训练样本，降低模型依赖共同话题的机会。他们还通过 STEL-Or-Content 让风格相似与内容相似直接竞争。专门训练的风格表征仍会在困难样例中受内容影响。[文献 14](https://aclanthology.org/2022.repl4nlp-1.26/)

这项研究的核心启发不是某个固定向量维数，而是训练和评估都要控制内容。普通语义 embedding 可能把“都在谈 Belgium”的两条消息放得很近，即使表达方式完全不同。

### 6.4 LISA：给神经表征的维度配上语言描述

LISA 使用 LLM 辅助生成风格属性标注，再训练属性预测并进行蒸馏。其基础表示包含 768 个对应语言化属性的分数，随后还研究了附加的度量学习层。输入为文本，输出为多维属性表示，因而可以检查哪些维度较高。[文献 15](https://aclanthology.org/2023.findings-emnlp.1020/)

它为“既要向量，又希望能解释变化”提供了重要路径。不过，论文也讨论了一些属性与内容边界不清的问题。基础属性向量与后续投影后的表示，其可解释方式并不完全相同。对短消息，未出现足够证据的属性也未必能被可靠判断。

### 6.5 StyleDistance：用内容接近的改写构造训练信号

StyleDistance 使用 LLM 构造接近同义、风格受控的文本，并围绕 40 种风格特征建立对比样本。其目的在于减少普通作者配对中风格和内容一起变化的问题，研究通过人工和自动评估检验了数据及表征。[文献 16](https://aclanthology.org/2025.naacl-long.436/)

这条路线表明，合成数据可以服务于测量工具的训练。但在合成改写上学得好，是否也能感知人类谈判短消息中的细微变化，仍然需要领域验证。

### 6.6 这些表征的取舍

| 方法 | 主要训练依据 | 更适合的用途 | 使用前最需要检查的事项 |
|---|---|---|---|
| LUAR | 同一作者的文本集合 | 作者习惯、作者验证、聚合表征 | 是否能反映同一个人的短时变化 |
| Wegmann 等的风格嵌入 | 作者关系加会话内容控制 | 内容干扰较小的风格相似性 | 换主题后是否仍按风格匹配 |
| LISA | 语言化风格属性及其预测 | 可解释的多维测量 | 属性含义、校准、短消息证据量 |
| StyleDistance | 内容接近且风格受控的合成对比样本 | 开放式风格距离 | 合成数据到真实对话的迁移 |

**共同限制。** 一个距离较小的结果不会直接告诉我们，谁受了谁的影响、改变发生在何时、为什么改变。上述方法可以成为认知模型的观测接口，但它们自身的目标通常不是解释人类的学习机制。

### 6.7 向量之外还要选择距离和观测窗口

余弦相似度比较向量的方向，欧氏距离还会受尺度影响。把句长、词频和属性概率直接拼接时，数值大的维度可能占主导，需要基于训练数据进行适当标准化。对预训练嵌入，则应优先检查其训练时使用的度量及归一化约定。

单条文本向量、同一个人最近几条消息的平均向量和全部历史平均向量，回答的问题不同。窗口变长可以提高稳定性，也会把状态变化变得平滑。窗口设计应与需要预测的时间尺度对应，不能看到哪种窗口产生预期结果后才决定主分析。

## 七、潜变量与风格转换：如何让内容和表达方式分别参与生成

### 7.1 内容码加风格条件

一个通用的生成设想是

$$
\mathbf c=E(u),\qquad
\hat u\sim p_\psi(u\mid\mathbf c,s_{\mathrm{target}}).
$$

编码器提取希望保留的信息，解码器根据内容码和目标风格生成文本。自编码器通过重建原文学习，VAE 为潜变量规定概率分布。额外的风格分类或对抗目标，用来鼓励生成符合属性，或减少内容码中的风格信息。

Hu 等把潜变量生成与属性控制结合，是早期可控文本生成的重要代表。Shen 等则在没有逐句配对改写的情况下，利用跨风格对齐训练风格转换模型。[文献 17](https://proceedings.mlr.press/v70/hu17e.html)、[文献 18](https://papers.nips.cc/paper_files/paper/2017/hash/2d2c8394e31101a261abf1784302bf75-Abstract.html)

### 7.2 为什么“解耦”不容易被证明

假如正式语料总在讨论法律，随意语料总在讨论游戏，模型很容易把主题也编码为风格。即使一个风格分类器无法从内容码预测原标签，也不能据此证明所有风格线索都已删除。分类器可能能力有限，另一些风格维度也可能没有标签。

因此，内容码与风格码应被看作由训练目标和数据约束出来的操作性分解。它是否符合我们想测量的表达差异，还需要改写对照、内容泄漏测试与下游验证。

### 7.3 删除、检索、生成

Li 等的 Delete, Retrieve, Generate 不要求先学习完全分离的潜空间。它先识别并删除属性相关表达，再从目标属性语料中检索表达材料，最后生成改写。其可解释性来自可以检查哪些片段被删除、借用了哪些目标表达。[文献 19](https://aclanthology.org/N18-1169/)

这适合风格标记较局部的任务，例如替换某些评价表达。若风格体现在整句句法或篇章组织，仅删除几个词就不充分，错误删除也可能损害内容。

### 7.4 回译与释义生成

Prabhumoye 等通过回译减弱原表达的一部分风格，再由风格相关生成器重建。Krishna 等的 STRAP 则将转换组织为释义生成与目标风格重建，借助自动产生的释义数据训练。[文献 20](https://aclanthology.org/P18-1080/)、[文献 21](https://aclanthology.org/2020.emnlp-main.55/)

两者的共同思想是先让原句的说法发生变化，再把目标表达倾向加回来。但所谓“中性释义”仍可能保留原风格，也可能已经改变意义。中间表示的作用需要通过实际样例和评估检验。

## 八、条件语言模型：直接学习在某种条件下怎么说

### 8.1 平行改写与监督微调

若有“同一内容、两种表达”的配对数据，可以训练 sequence-to-sequence 模型，输入原句与目标属性，输出目标句。GYAFC 提供非正式与正式文本的配对资料，是正式程度转换的重要资源。[文献 22](https://aclanthology.org/N18-1012/)

训练目标通常是提高目标文本的条件概率。

$$
\mathcal L_{\mathrm{SFT}}
=-\sum_{i=1}^{n}\log p_\psi(y_i\mid y_{\lt i},x,s).
$$

这里 $x$ 是输入，$s$ 是目标风格，$y$ 是参考输出。模型学的是“这个条件下，参考文本通常怎样写”。它能产生完整语言，也能计算候选句的概率。

比较两个候选句时，还要注意长度和候选空间。某个短句的整句概率较高，不一定表示模型单纯偏爱它的风格。若两句内容、长度差别很大，概率差包含了多个因素。

### 8.2 作者或伙伴条件化的对话模型

Li 等的 persona-based conversation model 把说话人信息，以及扩展模型中的说话人—受话人关系嵌入，对话生成时以这些表示为条件。它为“同一上下文面对不同的人可以产生不同回答”提供了直接的神经建模方式。[文献 23](https://aclanthology.org/P16-1094/)

这类架构可表示身份相关表达，但如果人物向量只是训练中学到的固定参数，就不能自动解释一次新暴露怎样改变当前表达。面向适应研究还需要在线更新，或把历史输入作为明确的条件。

### 8.3 Prefix-tuning 与 LoRA

Prefix-tuning 冻结基础模型，学习可被模型关注的连续前缀。LoRA 冻结原权重并训练低秩增量，常用形式为 $W'=W+BA$。它们降低了为某种任务或表达习惯适配模型时的可训练参数量。[文献 25](https://aclanthology.org/2021.acl-long.353/)、[文献 26](https://arxiv.org/abs/2106.09685)

这两篇提供通用训练技术，本身不等于人类风格适应理论。一个 LoRA 模块能够改变表达，也不能据此把模块参数直接解释成某个人的“迁移强度”。是否具有认知含义，取决于任务、更新规则和行为验证。

### 8.4 偏好优化

生成器还可以使用“同一输入下更喜欢哪个回答”的数据进行偏好优化。DPO 直接利用成对偏好调整模型相对于参考模型的输出倾向，是这类方法的代表。[文献 27](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html)

若用于风格，需要让偏好比较尽量只体现目标风格。否则更礼貌的回答也可能同时更长、更完整，模型会一起学习这些差别。偏好优化适合训练目标表达，不会自动揭示人受到某位伙伴影响的过程。

## 九、少样本模仿与个性化：从参考文本获得目标风格

### 9.1 自然语言提示与风格范例

提示式方法可以直接给出“更正式”“保留事实但更口语化”等指令，也可以提供目标作者的示例。Reif 等的 augmented zero-shot 方法使用通用改写示例，引导模型应用到新的目标风格，因而不必为每种目标风格都提供专门训练资料。[文献 24](https://aclanthology.org/2022.acl-short.94/)

优点是覆盖开放式风格，原型成本较低。问题是模型可能复制范例内容，结果对提示顺序、示例选择和基础模型变化敏感。通过提示要求模型“模仿 A”，然后看到它模仿 A，主要证明了指令执行能力，无法单独支持人类也采用同一种适应机制。

### 9.2 检索用户历史来个性化生成

LaMP 将个性化任务组织为当前请求加用户历史，比较不同的历史检索方法。RAGs to Style 进一步研究用风格嵌入检索用户示例，与按词语或上下文匹配进行比较。[文献 32](https://aclanthology.org/2024.acl-long.399/)、[文献 43](https://aclanthology.org/2024.personalize-1.11/)

这一路线把“历史中的哪些经历被重新取用”显式化，对项目中的近期输入与历史来源问题有启发。但个性化也可能依赖用户的知识、兴趣和常用主题，因此个性化质量不能完全替代风格评估。

### 9.3 TinyStyler：风格编码器加较小生成器

TinyStyler 将预训练风格嵌入接入约 800M 参数的 T5 生成器。训练先通过释义与风格向量重建原文，再用筛选的风格转换样本进行自蒸馏，使最终模型不必保留独立释义步骤。推理时输入原文和从目标参考文本获得的风格向量，输出改写。[文献 31](https://aclanthology.org/2024.findings-emnlp.781/)

这提供了与项目相近的组件分工。

```text
参考表达 → 风格编码器 → 目标风格向量
                              ↓
待表达的内容 ───────────→ 条件生成器 → 新文本
```

但其中还没有“P 听到 A 之后怎样更新、是否把影响带给 B”的认知学习规则。若借用该架构，适应模块需要另行规定并由人类数据约束。

## 十、推理时控制：调整词概率或隐藏层激活

### 10.1 PPLM、GeDi 与 FUDGE 的差别

三者都尝试在生成过程中加入控制信号，具体位置不同。

| 方法 | 控制信号进入哪里 | 直观解释 |
|---|---|---|
| **PPLM** | 根据属性目标的梯度调整生成时的内部状态 | 生成器往前走时，属性模型不断把它拉向目标 |
| **GeDi** | 通过类别条件语言模型得到属性相关概率，调整解码分布 | 比较候选延续更像哪种类别 |
| **FUDGE** | 预测当前不完整前缀最终满足属性的可能性，重加权候选 token | 选下一词时，考虑最终文本能否保持目标属性 |

这些方法提供不同的模块化控制接口。PPLM 不要求重新训练整个基础语言模型，GeDi 利用生成式判别，FUDGE 则明确使用未来属性预测。[文献 28](https://arxiv.org/abs/1912.02164)、[文献 29](https://aclanthology.org/2021.findings-emnlp.424/)、[文献 30](https://aclanthology.org/2021.naacl-main.276/)

它们适合控制可定义的属性。若属性模型把“问号多”误当成“礼貌”，生成器也会利用这一捷径。因此，控制效果需要由独立指标检验。

### 10.2 Activation steering：把风格方向加进隐藏层

Konen 等从特定风格文本的隐藏激活中计算向量，并在生成时加入隐藏层，以可调强度改变输出表达。[文献 33](https://aclanthology.org/2024.findings-eacl.52/)

这一类方法的共同示意形式为

$$
\mathbf h'_\ell=\mathbf h_\ell+\alpha\mathbf v_\ell.
$$

$\mathbf v_\ell$ 是第 $\ell$ 层的控制方向，$\alpha$ 控制干预强度。它与“把两条消息编码后求余弦相似度”的向量在用途上不同，前者用于干预生成，后者用于测量文本。

Zhang 等进一步通过对比激活进行个性化生成，尝试减少用户历史中内容信息对风格提取的干扰。这属于将作者历史转成生成控制量的较新路线。[文献 34](https://aclanthology.org/2025.acl-long.353/)

### 10.3 用可执行提示解释风格表示

Kim 与 Jurgens 的近期研究将风格表示解码为 style-eliciting prompts，即可以再次控制生成的自然语言提示。研究不仅检查描述是否合理，还检验恢复出的提示能否重新生成接近目标风格的文本。[文献 35](https://aclanthology.org/2026.findings-acl.2039/)

它为表征解释增加了行为性检验。一个描述若能指导生成相似表达，比仅给出漂亮的文字解释更有用途。不过，这种可执行性仍是在语言模型系统内验证的，不能直接作为人类内部表征的证据。

## 十一、认知与动态模型：解释风格如何随经验变化

### 11.1 误差驱动的隐式学习

Chang、Dell 与 Bock 的 dual-path 模型将句子产出与连接主义学习联系起来。模型利用事件意义和语言序列学习产出，当接触的结构偏离已有预测时，学习会改变后续表达倾向。它为句法启动作为持续学习的副产品提供了机制解释。[文献 36](https://www.kecl.ntt.co.jp/clip/member/chang/papers/chang%2Cdell%2Cbock.pdf)

**对项目的启发。** 同一学习系统的参数被不同伙伴共享时，局部经历可能影响面向其他伙伴的表达。是否真的如此，需要比较有无伙伴条件的架构。原研究关于句法的结果，也不能直接推广到所有风格维度。

### 11.2 层级贝叶斯：局部经验与共享规律之间的信息传递

Xu 与 Futrell 区分动词特定的句式倾向和跨动词的抽象倾向，通过贝叶斯更新预测后续句式选择。他们以模拟考察同动词优势、低频结构带来的较大变化，以及后续输入导致的差异衰减。[文献 37](https://arxiv.org/abs/2405.15964)

CHAI 将伙伴特定词义系统与群体预期分层建模，再结合 RSA 决定如何表达。它比较共享、分伙伴和部分汇聚，解释局部约定如何逐渐推广到其他交流伙伴。[文献 38](https://doi.org/10.1037/rev0000348)

二者分别提供了跨词语和跨伙伴泛化的机制参照。用于连续风格向量时，可以借用层级结构和后验预测的思想，但必须重新定义观测分布、伙伴层级和产出规则。这样的向量版本是我们需要检验的扩展，不是两篇论文已直接完成的结果。

### 11.3 记忆可及性：常用和刚用过的结构更容易取出

Reitter、Keller 与 Moore 在 ACT-R 框架下，用使用频率、新近程度和当前线索影响记忆激活，进而影响句法提取与产出。这条路线可以解释近期输入为什么更容易在后续表达中复用。[文献 39](https://homepages.inf.ed.ac.uk/keller/papers/cogsci_journal11.pdf)

需要区分两个结论。数据支持“近期经历权重更大”，并不直接等于支持“记忆搜索成本是原因”。指数折扣、持续学习和显式记忆模型都可能产生近期性。若要讨论提取成本，还需具有不同预测的模型比较，最好再有反应时或其他加工证据。

### 11.4 动态风格向量作为统一接口

下面是针对本项目的综合性建模示意，并非某篇文献的原模型。

$$
\mathbf s_{P,t+1}
=F\mathbf s_{P,t}+G\mathbf z_{A\to P,t}+\boldsymbol\epsilon_t,
$$

$$
\mathbf y_{P\to B,t}
\sim p(\mathbf y\mid\mathbf s_{P,t},\mathbf d_{P,B,t},\mathbf c_t).
$$

$\mathbf s_{P,t}$ 表示 P 的共享适应状态，$\mathbf d_{P,B,t}$ 表示面向 B 的特异状态，$\mathbf c_t$ 表示可观察语境。消息风格向量是观测量，内部状态则需要从历史中推断。

其价值在于把争论变成明确的模型差异。如果来自 A 的输入只能更新 A 特异状态，模型预测影响主要出现在 P 对 A 的产出。如果输入还会更新共享状态，模型就允许影响出现在 P 对 B 的产出。是否需要两个状态，应由留出数据中的预测和模型可区分性决定。

### 11.5 来源知识与实际表达可以分开

Ostrand 与 Ferreira 的人类句式研究比较总体输入及其在伙伴之间的分配，没有在相应实验中发现可靠的伙伴特异句法匹配。这个结果为总体暴露与来源分配的区分提供了行为参照，其适用范围是研究中的句式和任务。[文献 40](https://rostrand.github.io/papers/Ostrand_Ferreira.2019-JML.pdf)

对本项目，模型即使能够记住“谁怎么说”，也可能在产出时主要使用共享习惯。因而伙伴表征、学习更新和产出读取应该分别写清楚。仅观察 P 对 A/B 说得相似，无法判断他没有记住来源，还是记住了却没有用于此次表达。

## 十二、怎样评估：测量、生成和认知解释需要不同证据

### 12.1 风格表征的验证

STEL 将风格相似性组织为受控比较任务，使评估不必完全依赖作者分类准确率。后续 STEL-Or-Content 进一步让内容与风格竞争。[文献 13](https://aclanthology.org/2021.emnlp-main.569/)、[文献 14](https://aclanthology.org/2022.repl4nlp-1.26/)

针对新的对话语料，应至少检查以下情况。

| 检查 | 构造方式 | 希望看到的表现 |
|---|---|---|
| 同内容、不同风格 | 同一请求改为直接、间接或更正式的表达 | 能感受到目标维度变化 |
| 不同内容、相近风格 | 保持句式，替换地点和行动对象 | 不被任务名词主导 |
| 词语复用与类别复用 | 重复 maybe，与换用 perhaps 分开 | 清楚知道指标主要在量哪一层 |
| 长度敏感性 | 对不同长度的可比消息分别检验 | 报告短文本的不确定性 |
| 同人、不同情境 | 同一人的请求、解释、回应分别比较 | 不把交流目的全算作稳定个人风格 |
| 原文与遮盖 | 比较是否保留地名、身份等内容词 | 检查结论对主题线索的依赖 |

一个“纯风格”向量并没有可直接验证的万能标签。更实际的目标是证明指标对研究关心的差异敏感，并展示它对哪些干扰仍然敏感。

### 12.2 风格生成的验证

Mir 等区分风格变化强度、内容保持和自然度。Tikhonov 等进一步指出，一些常用自动指标会出现波动或被优化捷径利用，因此需要误差报告及适当的人类改写参照。[文献 41](https://aclanthology.org/N19-1049/)、[文献 42](https://aclanthology.org/D19-1406/)

例如，完全复制原句可能内容保持很好，却没有风格变化。生成一句目标风格很强但事实错误的话，也不能算成功。对话任务还需检查受话人、行动对象、承诺和否定是否保留。

若由 LLM 评分，最好使用独立的评估方式和人工可检查的对照，避免同一模型既定义风格、产生风格，又宣布任务成功。多种指标也应分别报告，便于判断取舍。

### 12.3 认知模型的验证

对解释人的模型，最重要的是预测真实人的后续行为，并让竞争机制在相同输入下接受比较。

- **预测比较。** 用同一套观测定义比较本人基线、伙伴特异学习和共享学习，在未用于拟合的游戏中预测。
- **时间约束。** 每次预测只允许使用此前的消息。窗口、标准化和降维的估计也要遵守训练—测试边界。
- **参数与模型恢复。** 用已知机制模拟数据，检查拟合能否找回机制。如果模型彼此无法区分，真实数据中的胜负也难作解释。
- **反事实预测。** 固定总暴露量，改变来源或顺序，检查模型如何预测。这能说明理论差别，真实因果判断仍需相应证据。
- **效应和不确定性。** 留出预测是否改善、改善有多大、是否由少数玩家驱动，比单个参数显著更完整。

## 十三、当前研究的主要进展与不足

### 13.1 从单一标签走向多维与开放式风格

多属性学习、风格嵌入和自然语言风格描述，让模型不必只围绕正式/非正式等二元任务工作。下一步困难在于维度是否可重复、是否能够跨语料校准，以及不同维度能否被独立改变。LISA 和 StyleDistance 分别代表可解释属性与内容受控度量学习的两条路径。[文献 15](https://aclanthology.org/2023.findings-emnlp.1020/)、[文献 16](https://aclanthology.org/2025.naacl-long.436/)

### 13.2 内容独立性有所改善，尚不能被默认

作者标签、语域标签和语义相近都可能提供捷径。控制配对、遮盖和释义是缓解手段，但各自引入新的假设。研究应说明在什么内容变化下表示仍然稳定，而不只给出一个总分。

### 13.3 可解释性开始接受可操作的检验

从解释属性分数，到把风格向量还原为能控制生成的提示，表征解释有了更多检验接口。接下来仍需区分描述可读、控制有效和心理机制正确这三种证据。[文献 35](https://aclanthology.org/2026.findings-acl.2039/)

### 13.4 生成能力与人类适应机制之间仍有距离

生成器可以模仿一个人的文本，却未必能解释一次接触如何改变另一个人的表达。认知研究需要更明确的经历表示、共享与特异状态、学习规则和行为比较。近期性、信息汇聚和交流目的可能同时作用，不能靠一次模仿成功加以区分。

### 13.5 短消息、跨伙伴和非英语情境需要单独验证

长篇作者风格与短消息中的一次表达，证据量不同。本文代表作也大量使用英语及特定网络语料，其结果不能无条件推广到中文语气词、语序和中英混合表达。对于多方互动，还需区分当前伙伴输入、其他伙伴输入和本人之前的产出。

## 十四、对 Beyond the Dyad 的具体判断

本节内容已并入[方法文档](02-method.zh.md)：测量与学习的整体流程见第 1.2 节，三条测量路线及选择见第 2.1 节，数据带来的约束见第 2.2、2.4 节，贡献形式见第 6.5 节。语料统计见[数据文档](01-data-and-pilot.zh.md)。

## 十五、建议的阅读顺序

| 阅读目的 | 优先文献 | 读完应当能回答的问题 |
|---|---|---|
| 建立总体方法地图 | Jin 等，2022［1］ | 风格迁移为什么有多种训练设定？ |
| 从自然对话测量对齐 | Doyle 与 Frank，2016［7］ | 怎么区分词语复用、类别对齐与本人基线？ |
| 理解内容泄漏 | Wegmann 等，2022［14］ | 为什么能认作者的向量未必在测纯风格？ |
| 选择可解释向量 | Patel 等，2023［15］ | 文本怎样被转成有描述的多维分数？ |
| 选择开放式风格距离 | Patel 等，2025［16］ | 内容受控的配对如何改变表示学习？ |
| 理解共享与来源特异学习 | Hawkins 等，2023［38］ | 局部约定为什么、何时会推广？ |
| 比较近期性的机制 | Reitter 等，2011［39］及 Xu 与 Futrell，2024［37］ | 记忆可及性与持续学习能否产生相似模式？ |
| 为生成扩展选接口 | Horvitz 等，2024［31］ | 风格编码器怎样接到较小的生成器？ |

## 参考文献

以下优先链接论文出版页、作者稿或作者维护的资源。年份采用文献所列正式发表年份；预印本另作说明。

1. Jin, D., Jin, Z., Hu, Z., Vechtomova, O., & Mihalcea, R. (2022). [Deep Learning for Text Style Transfer: A Survey](https://doi.org/10.1162/coli_a_00426). *Computational Linguistics, 48*(1), 155–205. **总体方法综述。**
2. Biber, D. (1988). [Variation across Speech and Writing](https://doi.org/10.1017/CBO9780511621024). Cambridge University Press. **多维度语域分析。**
3. Stamatatos, E. (2009). [A Survey of Modern Authorship Attribution Methods](https://doi.org/10.1002/asi.21001). *Journal of the American Society for Information Science and Technology, 60*(3), 538–556. **计算文体学与作者识别。**
4. Burrows, J. (2002). [‘Delta’: a Measure of Stylistic Difference and a Guide to Likely Authorship](https://doi.org/10.1093/llc/17.3.267). *Literary and Linguistic Computing, 17*(3), 267–287. **标准化词频距离。**
5. Ireland, M. E., & Pennebaker, J. W. (2010). [Language style matching in writing: Synchrony in essays, correspondence, and poetry](https://doi.org/10.1037/a0020386). *Journal of Personality and Social Psychology, 99*(3), 549–571. **功能词匹配。**
6. Danescu-Niculescu-Mizil, C., Lee, L., Pang, B., & Kleinberg, J. (2012). [Echoes of Power: Language Effects and Power Differences in Social Interaction](https://www.cs.cornell.edu/~cristian/Echoes_of_power_files/echoes_of_power.pdf). *WWW*. **有方向的语言协调。**
7. Doyle, G., & Frank, M. C. (2016). [Investigating the Sources of Linguistic Alignment in Conversation](https://aclanthology.org/P16-1050/). *ACL*. **WHAM、词汇与类别对齐。**
8. Halvani, O., & Graner, L. (2021). [POSNoise: An Effective Countermeasure Against Topic Biases in Authorship Analysis](https://arxiv.org/abs/2005.06605). *ARES*. DOI 10.1145/3465481.3470050. **内容遮盖；预印本初版为 2020 年。**
9. Eisenstein, J., Ahmed, A., & Xing, E. P. (2011). [Sparse Additive Generative Models of Text](https://research.google/pubs/sparse-additive-generative-models-of-text/). *ICML*. **多因素稀疏概率文本模型。**
10. Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013). [A computational approach to politeness with application to social factors](https://aclanthology.org/P13-1025/). *ACL*. **礼貌特征与监督预测。**
11. Kang, D., & Hovy, E. (2021). [Style is NOT a single variable: Case Studies for Cross-Stylistic Language Understanding](https://aclanthology.org/2021.acl-long.185/). *ACL-IJCNLP*. **多属性风格理解、XSLUE。**
12. Rivera-Soto, R. A., Miano, O. E., Ordonez, J., Chen, B. Y., Khan, A., Bishop, M., & Andrews, N. (2021). [Learning Universal Authorship Representations](https://aclanthology.org/2021.emnlp-main.70/). *EMNLP*. **LUAR。** [作者代码](https://github.com/LLNL/LUAR)。
13. Wegmann, A., & Nguyen, D. (2021). [Does It Capture STEL? A Modular, Similarity-based Linguistic Style Evaluation Framework](https://aclanthology.org/2021.emnlp-main.569/). *EMNLP*. **风格表征评估。**
14. Wegmann, A., Schraagen, M., & Nguyen, D. (2022). [Same Author or Just Same Topic? Towards Content-Independent Style Representations](https://aclanthology.org/2022.repl4nlp-1.26/). *RepL4NLP*. **内容受控的风格嵌入。** [作者代码](https://github.com/nlpsoc/Style-Embeddings)。
15. Patel, A., Rao, D., Kothary, A., McKeown, K., & Callison-Burch, C. (2023). [Learning Interpretable Style Embeddings via Prompting LLMs](https://aclanthology.org/2023.findings-emnlp.1020/). *Findings of EMNLP*. **LISA、StyleGenome。**
16. Patel, A., Zhu, J., Qiu, J., Horvitz, Z., Apidianaki, M., McKeown, K., & Callison-Burch, C. (2025). [StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples](https://aclanthology.org/2025.naacl-long.436/). *NAACL*. **合成平行样本与风格距离。**
17. Hu, Z., Yang, Z., Liang, X., Salakhutdinov, R., & Xing, E. P. (2017). [Toward Controlled Generation of Text](https://proceedings.mlr.press/v70/hu17e.html). *ICML*. **潜变量与属性控制。**
18. Shen, T., Lei, T., Barzilay, R., & Jaakkola, T. (2017). [Style Transfer from Non-Parallel Text by Cross-Alignment](https://papers.nips.cc/paper_files/paper/2017/hash/2d2c8394e31101a261abf1784302bf75-Abstract.html). *NeurIPS*. **非平行风格转换。**
19. Li, J., Jia, R., He, H., & Liang, P. (2018). [Delete, Retrieve, Generate: a Simple Approach to Sentiment and Style Transfer](https://aclanthology.org/N18-1169/). *NAACL*. **编辑与检索生成。**
20. Prabhumoye, S., Tsvetkov, Y., Salakhutdinov, R., & Black, A. W. (2018). [Style Transfer Through Back-Translation](https://aclanthology.org/P18-1080/). *ACL*. **回译式风格转换。**
21. Krishna, K., Wieting, J., & Iyyer, M. (2020). [Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://aclanthology.org/2020.emnlp-main.55/). *EMNLP*. **STRAP。** [作者代码](https://github.com/martiansideofthemoon/style-transfer-paraphrase)。
22. Rao, S., & Tetreault, J. (2018). [Dear Sir or Madam, May I Introduce the GYAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer](https://aclanthology.org/N18-1012/). *NAACL*. **正式程度配对语料。**
23. Li, J., Galley, M., Brockett, C., Spithourakis, G., Gao, J., & Dolan, B. (2016). [A Persona-Based Neural Conversation Model](https://aclanthology.org/P16-1094/). *ACL*. **身份与交互条件化生成。**
24. Reif, E., Ippolito, D., Yuan, A., Coenen, A., Callison-Burch, C., & Wei, J. (2022). [A Recipe For Arbitrary Text Style Transfer with Large Language Models](https://aclanthology.org/2022.acl-short.94/). *ACL*. **提示式开放风格转换。**
25. Li, X. L., & Liang, P. (2021). [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/). *ACL-IJCNLP*. **通用参数高效适配。**
26. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *ICLR*. **通用低秩适配。**
27. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html). *NeurIPS*. **通用偏好优化。**
28. Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., Yosinski, J., & Liu, R. (2020). [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](https://arxiv.org/abs/1912.02164). *ICLR*. **PPLM。**
29. Krause, B., et al. (2021). [GeDi: Generative Discriminator Guided Sequence Generation](https://aclanthology.org/2021.findings-emnlp.424/). *Findings of EMNLP*. **生成式判别控制。**
30. Yang, K., & Klein, D. (2021). [FUDGE: Controlled Text Generation With Future Discriminators](https://aclanthology.org/2021.naacl-main.276/). *NAACL*. **前缀属性预测控制。**
31. Horvitz, Z., Patel, A., Singh, K., Callison-Burch, C., McKeown, K., & Yu, Z. (2024). [TinyStyler: Efficient Few-Shot Text Style Transfer with Authorship Embeddings](https://aclanthology.org/2024.findings-emnlp.781/). *Findings of EMNLP*. **嵌入条件化的小型生成器。**
32. Salemi, A., Mysore, S., Bendersky, M., & Zamani, H. (2024). [LaMP: When Large Language Models Meet Personalization](https://aclanthology.org/2024.acl-long.399/). *ACL*. **用户历史与个性化基准。**
33. Konen, K., Jentzsch, S., Diallo, D., Schütt, P., Bensch, O., El Baff, R., Opitz, D., & Hecking, T. (2024). [Style Vectors for Steering Generative Large Language Models](https://aclanthology.org/2024.findings-eacl.52/). *Findings of EACL*. **激活控制。**
34. Zhang, J., Liu, Y., Wang, W., Liu, Q., Wu, S., Wang, L., & Chua, T.-S. (2025). [Personalized Text Generation with Contrastive Activation Steering](https://aclanthology.org/2025.acl-long.353/). *ACL*. **StyleVector、个性化激活控制。**
35. Kim, J., & Jurgens, D. (2026). [Interpreting Style Representations via Style-Eliciting Prompts](https://aclanthology.org/2026.findings-acl.2039/). *Findings of ACL*. **可执行的风格解释；近期方向。**
36. Chang, F., Dell, G. S., & Bock, K. (2006). [Becoming Syntactic](https://www.kecl.ntt.co.jp/clip/member/chang/papers/chang%2Cdell%2Cbock.pdf). *Psychological Review, 113*(2), 234–272. **误差驱动句子产出与隐式学习。**
37. Xu, W., & Futrell, R. (2024). [A hierarchical Bayesian model for syntactic priming](https://arxiv.org/abs/2405.15964). arXiv:2405.15964. **动词特异与共享句式知识；此处链接预印本。**
38. Hawkins, R. D., Franke, M., Frank, M. C., Goldberg, A. E., Smith, K., Griffiths, T. L., & Goodman, N. D. (2023). [From Partners to Populations: A Hierarchical Bayesian Account of Coordination and Convention](https://doi.org/10.1037/rev0000348). *Psychological Review, 130*(4), 977–1016. **CHAI、伙伴与群体学习。** [作者稿](https://cocosci.princeton.edu/papers/hawkinspartners.pdf)。
39. Reitter, D., Keller, F., & Moore, J. D. (2011). [A Computational Cognitive Model of Syntactic Priming](https://homepages.inf.ed.ac.uk/keller/papers/cogsci_journal11.pdf). *Cognitive Science, 35*(4), 587–637. **ACT-R 与记忆可及性。**
40. Ostrand, R., & Ferreira, V. S. (2019). [Repeat after us: Syntactic alignment is not partner-specific](https://rostrand.github.io/papers/Ostrand_Ferreira.2019-JML.pdf). *Journal of Memory and Language, 108*, 104037. **总体暴露与伙伴来源的人类对照。**
41. Mir, R., Felbo, B., Obradovich, N., & Rahwan, I. (2019). [Evaluating Style Transfer for Text](https://aclanthology.org/N19-1049/). *NAACL*. **风格、内容、自然度的分别评估。**
42. Tikhonov, A., Shibaev, V., Nagaev, A., Nugmanova, A., & Yamshchikov, I. P. (2019). [Style Transfer for Texts: Retrain, Report Errors, Compare with Rewrites](https://aclanthology.org/D19-1406/). *EMNLP-IJCNLP*. **指标可靠性与改写参照。**
43. Neelakanteswara, A., Chaudhari, S., & Zamani, H. (2024). [RAGs to Style: Personalizing LLMs with Style Embeddings](https://aclanthology.org/2024.personalize-1.11/). *PERSONALIZE Workshop*. **风格驱动的历史检索；探索性工作。**
44. Boyd, R. L., Ashokkumar, A., Seraj, S., & Pennebaker, J. W. (2022). [The Development and Psychometric Properties of LIWC-22](https://liwc.app/help/psychometrics-manuals). The University of Texas at Austin. **词典式语言分析与指标说明。**

