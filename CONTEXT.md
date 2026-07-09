# EDRE (Evidence-Driven Research Engine)

位于 search_agent 检索层之上的研究编排引擎。
围绕用户问题规划待验证的知识点，多轮收集证据、评估完整性，直至达到可高置信度回答的状态。
核心状态是 EvidenceClaim，而非 Query 或 Document。

## Language

**EvidenceClaim**:
一个需要被验证或证实的具体知识断言（hypothesis），例如"Claude Code 使用 Prompt Cache，因此减少 Token 消耗"。
不是一个 Topic（如"Prompt Cache"）。是全系统最核心的数据结构。
_Avoid_: Topic, Claim（单独用时）, Fact

**support（有符号）**:
第二层 LLM 对一对 (文档, claim) 给出的结论性打分 `∈ [-1, 1]`。正=支持该 claim，负=反驳/证伪该 claim。
_Avoid_: relevance（相关性是第一层 reranker 的事，不是 support）

**confidence**:
EvidenceClaim 的结论性程度，取其所有证据中最具决定性的 support（按绝对值最大），是该 claim 进度的唯一真相来源。符号决定走向 VERIFIED 还是 REFUTED。v1 不特殊处理"证据互相矛盾"。
_Avoid_: score（泛指时）

**status**:
EvidenceClaim 的进度视图，全部由 confidence + search_attempts 派生、只读、不独立存储：
进行中 `NOT_STARTED / PARTIAL`；终止 `VERIFIED`（强正）/ `REFUTED`（强负，证伪是成功发现）/ `ABANDONED`（预算耗尽仍不决定性）。
_Avoid_: state；不要把 REFUTED 混同为 ABANDONED

**Evidence Plan**:
Planner 针对一个 task 产出的 EvidenceClaim 集合，代表"要回答此问题需验证哪些知识点"。
_Avoid_: Query Plan, Search Plan

**importance**:
EvidenceClaim 的二元重要度 `关键 / 可选`。停止决策只要求"所有关键 claim 都 VERIFIED"。
_Avoid_: priority, weight（不用连续权重）

**Coverage**:
纯展示用的进度比例 `已 VERIFIED 数 / 总数`。不参与停止决策（决策由"所有关键 claim 是否 VERIFIED"驱动）。
_Avoid_: 把 Coverage 当作带阈值的决策量

**DONE / EXHAUSTED**:
研究循环的两种终止态。DONE = 所有关键 claim 已解决（VERIFIED 或 REFUTED）；EXHAUSTED = 撞 MaxLoop 或无进展而退出，仍有关键 claim 未解决（尽力未成，不得伪装成 DONE）。
_Avoid_: 用单一 finished 布尔混淆两者

**research_summary**:
v1 的最终输出之一：**确定性的结果快照**（终止态、各 claim verdict 计数、coverage、loop_count），纯拼装、无 LLM。
不是对 task 的成文叙述答案（那是 Answer Synthesis，属 R7）。
_Avoid_: Answer, 叙述式答案, LLM 合成总结

**search_many adapter**:
EDRE 调用检索的唯一入口：输入多个 query，返回带 query 归属、已 URL 去重的文档集合。复用 search_agent 的 fan-out 与去重，跳过其 decompose 与 query 合并式 answer 输出。
_Avoid_: search_agent 子图（EDRE 不直接调用完整子图）

**文档规范化（Normalization）**:
把每篇 reranker 幸存文档蒸馏成 dense、保留 citation、带冲突标注的事实片段，作为下游 claim 打分的 grounding。改造复用现有 consolidate 的 LLM 规范化能力，但改为逐文档粒度（而非一个 query 合并成一坨）。
_Avoid_: consolidate/answer 合成（EDRE 不产 query 合并式 answer，规范化是逐文档的）
