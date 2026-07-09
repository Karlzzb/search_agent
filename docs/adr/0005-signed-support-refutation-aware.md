# 有符号 support：引擎能证伪 claim，而非只能"未验证"

## 背景

Planner 产出的 EvidenceClaim 是 hypothesis（可能为假的断言）。
若 confidence 只测"支持度"（support ∈ [0,1]），那么"证据反驳了 claim"和"没查到证据"会塌成同一个低 confidence，最终都判 ABANDONED / EXHAUSTED。
但对 evidence-driven 引擎，"断言为假"是一个成功的发现，不该谎报成"尽力未成"。

## 决定

第二层 LLM 的 (文档, claim) 打分从 `[0,1]` 拓宽为**有符号** `support ∈ [-1,1]`（负=反驳）。
claim 的终止态由此变为三种：`VERIFIED`（强正）/ `REFUTED`（强负，可报告的成功发现）/ `ABANDONED`（预算耗尽仍不决定性）。
ResearchOutput 必须区分 REFUTED 与 ABANDONED。

## 权衡

近零成本：不引入新组件、新公式，仅把已有那次 LLM 调用的打分范围拓宽，符号带出证伪语义。
v1 不特殊处理"同一 claim 证据互相矛盾"（取绝对值最大的 support 决定走向），该边界留待后续。
