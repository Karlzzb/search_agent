# v1：薄 Controller + 固定 claim 集，收敛 7 动作空间

## 背景

PRD 第 10 节把 Controller 描述为"整个系统的大脑"，支持 7 个动作 `FINISH / REWRITE_QUERY / GENERATE_QUERY / SEARCH_MORE / REPLAN / CHANGE_PROVIDER / STOP`。
逐个分析后，多数动作要么重叠、要么属于其他组件的职责。

## 决定

v1 大幅收敛：

- `FINISH / STOP` 不是 Controller 主动选择，而是由停止条件计算出的终止态 **DONE / EXHAUSTED**。
- `GENERATE_QUERY / REWRITE_QUERY / SEARCH_MORE` 塌成单一"继续检索"；改写还是换新 query 由 **Query Generator 内部策略**决定，Controller 不指挥。
- `CHANGE_PROVIDER` 下沉为 **search_many adapter 的故障转移**，非研究层决策。
- `REPLAN`（循环中增删 claim）**推迟到后续 Phase**；v1 **固定 claim 集**。

结果：v1 Controller 只是一个停止条件评估器，决策空间 = `{CONTINUE + 自动终止}`，不是 LLM 决策体。
v1 的智能集中在 Planner + Query Generator + Evaluation。

## 权衡

固定 claim 集的代价：若 Planner 产出一个查不到的 claim，循环会在它上面 EXHAUST，输出如实标注该 claim 未验证——v1 可接受。
换来的是恒定的 coverage 分母、完全可解释的收敛、以及大幅简化的状态模型。
REPLAN 带来的动态分母与收敛性复杂度不进 v1。
