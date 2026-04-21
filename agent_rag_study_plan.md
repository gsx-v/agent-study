# Agent & RAG 求职项目计划

> 目标: Agentic RAG + 多Agent协作系统 | 开始: 2026-04-21 | 周期: 5周
> 技术栈: LangGraph, LangChain, LlamaIndex, Neo4j, FastAPI, Chroma, RAGAS, Docker

---

## 第1周：基础 + Agentic RAG 入门

### Day1 (4/21) LangGraph复习+环境搭建
- [ ] 复习 StateGraph/Node/Edge/ConditionalEdge/ToolNode
- [ ] 阅读官方Quickstart: langchain-ai.github.io/langgraph/
- [ ] 搭建环境: venv + langchain,langgraph,llama-index,chromadb,fastapi
- [ ] 跑通最简单graph, repo初始化

### Day2 (4/22) 基础RAG流水线
- [ ] LlamaIndex: SimpleDirectoryReader + VectorStoreIndex
- [ ] 或LangChain: RecursiveCharacterTextSplitter + Chroma
- [ ] 实现: PDF加载→分块→嵌入→存储→检索→生成
- [ ] 基础RAG可运行

### Day3 (4/23) 官方Agentic RAG教程
- [ ] 精读 docs.langchain.com/oss/python/langgraph/agentic-rag
- [ ] 理解CRAG: 检索→评分→路由
- [ ] 逐行跟跑教程代码

### Day4 (4/24) 参考agentic-rag-for-dummies
- [ ] 克隆 github.com/GiovanniPasq/agentic-rag-for-dummies
- [ ] 跑通并理解模块化架构(core/db/config)

### Day5 (4/25) Web Search工具
- [ ] 阅读qdrant.tech Agentic RAG教程
- [ ] 集成Tavily/WebSearch工具
- [ ] 实现检索失败→自动Web搜索降级

### Day6 (4/26) 自研Agentic RAG v1
- [ ] 用自己代码实现: 提问→路由→检索→评分→相关?生成:重写→再检索→生成
- [ ] 不照抄,用自己的理解重写

### Day7 (4/27) 休息+整理
- [ ] 整理笔记, 画Mermaid流程图, 确认Neo4j Docker等

---

## 第2周：完善Agentic RAG + 工程化

### Day8 (4/28) GraphRAG基础
- [ ] 阅读Microsoft GraphRAG: microsoft.com/en-us/research/project/graphrag/
- [ ] 理解: 实体抽取→知识图谱→社区检测→社区摘要→全局检索
- [ ] 安装Neo4j Docker, 跑通github.com/microsoft/graphrag quickstart

### Day9 (4/29) LlamaIndex GraphRAG
- [ ] 精读docs.llamaindex.ai GraphRAG_v2教程
- [ ] PropertyGraphIndex + Neo4jPropertyGraphStore
- [ ] 文档→实体关系抽取→存入Neo4j

### Day10 (4/30) 混合检索: 向量+图谱
- [ ] 向量检索(局部) + 图谱检索(全局) + RRF融合排序
- [ ] Agentic RAG v2支持双检索

### Day11 (5/1) FastAPI后端+流式输出
- [ ] FastAPI封装REST API + SSE流式输出
- [ ] 接口: /query, /upload, /history

### Day12 (5/2) RAGAS评估+LangSmith
- [ ] RAGAS指标: Faithfulness/AnswerRelevancy/ContextPrecision
- [ ] 构建10-20问答对评估集, 记录baseline
- [ ] 接入LangSmith追踪

### Day13 (5/3) 前端+持久化
- [ ] Gradio/Streamlit对话界面, 显示来源和引用
- [ ] SqliteSaver/PostgresSaver持久化

### Day14 (5/4) 项目1收尾
- [ ] 写README(架构图/技术选型/评估数据), 提交Git

---

## 第3周：多Agent协作系统

### Day15 (5/5) Multi-Agent模式学习
- [ ] 阅读github.com/langchain-ai/langgraph-supervisor-py
- [ ] 理解Supervisor模式, 跑通官方示例

### Day16 (5/6) 设计架构
- [ ] 设计4个Agent: Supervisor/Researcher/Coder/Reviewer
- [ ] 定义prompt/工具/状态, 画架构图

### Day17 (5/7) Researcher Agent
- [ ] 工具: Tavily + Agentic RAG(复用项目1)
- [ ] 实现LangGraph子图, 可独立运行

### Day18 (5/8) Coder+Reviewer Agent
- [ ] Coder: PythonREPL/文件写入 | Reviewer: 代码分析/测试执行
- [ ] 各自实现为子图, 可独立运行

### Day19 (5/9) Supervisor+整合
- [ ] Supervisor: 分析→路由→收集→决定下一步/结束
- [ ] 组装4个Agent为完整graph, 共享State

### Day20 (5/10) 高级特性
- [ ] Human-in-the-loop: interrupt_before关键节点
- [ ] 错误恢复 + 并行执行(Send API)

### Day21 (5/11) 项目2收尾
- [ ] 测试调试, 写README

---

## 第4周：整合+前端+部署

### Day22 (5/12) 整合两项目
- [ ] Agentic RAG作为Researcher核心工具
- [ ] 统一FastAPI后端, 端到端流程跑通

### Day23 (5/13) 前端完善
- [ ] React/Gradio: 对话+来源展示+Agent步骤可视化+人工确认

### Day24 (5/14) Docker部署
- [ ] Dockerfile + docker-compose.yml一键启动

### Day25 (5/15) 对比评估
- [ ] RAGAS对比: 纯向量RAG vs GraphRAG vs Agentic RAG
- [ ] 记录延迟/token/质量, 生成对比表

### Day26 (5/16) Demo+博客
- [ ] 录制3-5分钟演示视频
- [ ] 写技术博客: 架构/踩坑/评估

### Day27-28 (5/17-18) 最终打磨
- [ ] README完善, 代码清理
- [ ] 确保git clone + docker-compose up即可运行

---

## 核心资源

| 资源 | 链接 |
|------|------|
| LangGraph文档 | langchain-ai.github.io/langgraph/ |
| Agentic RAG教程 | docs.langchain.com/oss/python/langgraph/agentic-rag |
| agentic-rag-for-dummies | github.com/GiovanniPasq/agentic-rag-for-dummies |
| LangGraph Supervisor | github.com/langchain-ai/langgraph-supervisor-py |
| LlamaIndex GraphRAG | docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/ |
| Microsoft GraphRAG | github.com/microsoft/graphrag |
| GenAI_Agents教程集 | github.com/NirDiamant/GenAI_Agents |
| Qdrant Agentic RAG | qdrant.tech/documentation/tutorials-build-essentials/agentic-rag-langgraph/ |

---

## 每日提醒
1. 代码每天提交Git
2. 核心概念要能用自己的话解释
3. 遇到卡点限时1小时, 超过就跳过或简化
4. 记录决策理由: "为什么选X而不是Y" 比代码本身更重要
5. API Key用.env文件, 绝不硬编码
