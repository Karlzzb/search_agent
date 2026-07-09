# EDRE 用单张扁平 LangGraph，v1 放弃"状态可恢复"

## 背景

EDRE 的研究循环需要一个编排载体。现有 search_agent 已是 LangGraph + Langfuse。
PRD 非功能要求列了"状态可恢复"。

## 决定

- EDRE 是**一张扁平的 LangGraph**，与现有代码同构，但**不**采用"父图嵌套 search_agent 子图"的结构。
- `search_many` adapter 只是图中一个**普通节点**（函数调用），不是被 compile 进来的子图；这与 ADR-0001 一致（EDRE 本就不调完整子图）。
- 循环用一条回边实现，`recursion_limit` 兼作 MaxLoop 安全网。
- **不接 checkpointer、不做续跑**，`ResearchState` 只存在于内存。
- 因此 v1 **有意识地放弃** PRD 非功能要求中的"状态可恢复"。

## 权衡

嵌套子图会引入编译/包裹开销且与"EDRE 不调完整子图"的决定冲突，故拍平。
续跑（LangGraph checkpointer + 持久后端）对多轮烧钱的循环有真实价值，但属于重型基础设施；v1 先不背，等有实际需求再加，不影响上层逻辑。
