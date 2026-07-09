# 评估分数拆成两层，废弃 PRD 的"四项相加" Final Score

## 背景

PRD 第 8 节定义 `Final Score = Task Match + Query Match + Claim Support + Coverage`。
但这四项粒度不一致：Task Match / Query Match 是每篇文档的分，Claim Support 是"文档 × claim"矩阵，Coverage 是整个 ResearchState 的全局量。
把每篇文档的分与全局研究进度相加是范畴错误，得数无意义；且矩阵无法直接进标量加法。

## 决定

按其服务的决策拆成两层，Coverage 不参与文档打分：

- **文档门控分** `doc_relevance = w1·TaskMatch + w2·QueryMatch + w3·maxClaimSupport(doc)`，用于决定单篇文档去留与排序（Precision First 闸门）。
- **Claim Support 矩阵** `claim_support[doc][claim]` 单独留存，喂给 Evidence Updater 更新每个 claim 的 confidence。
- **Coverage** 由所有 claim 的 confidence 按 importance 加权聚合，只作为 Controller 的停止决策输入。

PRD 的单一 Final Score 作废。

## 权衡

四项相加实现上更"统一"，但语义不成立且不可解释。
两层模型让每个分各司其职（门控 / 驱动 confidence / 驱动停止），符合非功能要求的"可解释"与"Score 可配置"。
