# ### Day1 (4/21) LangGraph复习+环境搭建
# - [ ] 复习 StateGraph/Node/Edge/ConditionalEdge/ToolNode
# - [ ] 阅读官方Quickstart: langchain-ai.github.io/langgraph/
# - [ ] 搭建环境: venv + langchain,langgraph,llama-index,chromadb,fastapi
# - [ ] 跑通最简单graph, repo初始化

# 一个简单的HelloWorld示例-使用langraph实现
from langgraph.graph import StateGraph, MessagesState, START, END

# 定义函数模拟LLM
def mock_llm(state: MessagesState):
    return {"messages": [{"role": "assistant", "content": "Hello World"}]}

# 创建图
graph = StateGraph(MessagesState)
# 添加节点
graph.add_node(mock_llm)
# 添加边
graph.add_edge(START, "mock_llm")
# 添加边
graph.add_edge("mock_llm", END)
# 编译图
app = graph.compile()
# 运行图
result = app.invoke({"messages": [{"role": "user", "content": "Hello"}]})
print(result)
