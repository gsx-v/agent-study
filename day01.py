# ### Day1 (4/21) LangGraph复习+环境搭建
# - [ ] 复习 StateGraph/Node/Edge/ConditionalEdge/ToolNode
# - [ ] 阅读官方Quickstart: langchain-ai.github.io/langgraph/
# - [ ] 搭建环境: venv + langchain,langgraph,llama-index,chromadb,fastapi
# - [ ] 跑通最简单graph, repo初始化

# 一个简单的HelloWorld示例-使用langraph实现
from langchain import messages
# 导入langgraph的状态图，消息状态，开始，结束节点
from langgraph.graph import StateGraph, MessagesState, START, END

# # 定义函数模拟LLM
# def mock_llm(state: MessagesState):
#     return {"messages": [{"role": "assistant", "content": "Hello World"}]}

# 导入大模型，这里使用deepseek
from langchain_deepseek import ChatDeepSeek
# 导入环境变量
from dotenv import load_dotenv
load_dotenv()
import os
# 导入工具
from langchain.tools import tool
# 定义状态
from langchain.messages import AnyMessage
# 定义类型字典
from typing_extensions import TypedDict, Annotated
import operator
# 导入SystemMessage，消息模板。
from langchain.messages import SystemMessage
# 导入ToolMessage用于记录工具执行结果
from langchain.messages import ToolMessage
# 导入类型Literal
from typing import Literal

# 定义模型
model = ChatDeepSeek(
    model = "deepseek-chat",
    api_key = os.getenv("DeepSeek_API_KEY"),
    base_url = os.getenv("Base_URL")
)


# 创建工具,用于两数相乘
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# 创建工具,用于两数相加
@tool
def add(a: int, b: int) -> int:
    """Add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

# 创建工具列表
tools = [multiply, add]
# 绑定工具到模型
model_with_tools = model.bind_tools(tools)
# 定义工具字典，方便后续查找工具
tools_by_name = {tool.name: tool for tool in tools}

# 定义状态，类型为TypedDict，且Annotated类型可添加，使用operator添加
# Annotated是 Python 类型系统中一个非常强大的工具，它允许我们在类型注解中附加元数据。
# 基础类型：list[AnyMessage]（一个包含任意消息的列表）
# 元数据：operator.add（定义了如何合并这个字段）
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# 定义模型节点，用于调用LLM，并决定是否调用某个工具
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages":[
            # [SystemMessage] + 历史消息
            # 把系统指令放在对话历史前面一起传给模型
            # 模型看到的是：系统指令 + 用户消息 + 之前的工具结果（如果有）
            model_with_tools.invoke(
                [
                SystemMessage(
                    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                )
            ]
            + state["messages"]
            )
        ],
        # 记录 LLM 被调用了几次，方便调试和限制（比如防止无限循环）
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# 定义工具节点,以下为手写，可以使用ToolNode直接识别工具
def tool_node(state: dict):
    """Performs the tool call"""
    
    # 创建字典
    result = []
    # 取最后一条消息，也就是 LLM 刚输出的 AIMessage,
    # 只有 AIMessage 才会有 tool_calls 字段（模型决定调用工具时生成）
    # 然后遍历其中的每个工具调用
    for tool_call in state["messages"][-1].tool_calls:
        # 获取工具
        tool = tools_by_name[tool_call["name"]]
        # 执行工具
        observation = tool.invoke(tool_call["args"])
        # 将工具执行结果添加到消息列表中
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    
    # return {"messages": result}
    # 返回所有工具结果，通过 operator.add 追加到消息列表
    # 下次 LLM 看到的消息序列就是：用户消息 → AIMessage(我要调工具) 
    # → ToolMessage(结果是15) → LLM 据此生成最终回答
    return {"messages": result}

# 定义终端逻辑
# 条件边缘函数用于根据LLM是否调用工具调用，将节点或端路由到工具节点。
# 返回值类型标注，表示这个函数只能返回两种字符串："tool_node" 或 END
def should_continue(state: MessagesState)->Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    
    messages = state["messages"]
    # 提取最后一条信息
    last_message = messages[-1]
    
    # 检查最后一条消息是否包含工具调用
    if last_message.tool_calls:
        return "tool_node"
    
    return END

# 构建并编译图
# 创建图
graph = StateGraph(MessagesState)
# 添加节点
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)
# 添加边
graph.add_edge(START, "llm_call")
# 添加条件边
graph.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
# 添加边
graph.add_edge("tool_node", "llm_call")
# 编译图
app = graph.compile()
# 画出示意图
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
# 运行图
# 导入用户系统消息模板
from langchain.messages import HumanMessage
messages = [HumanMessage(content="计算5乘以4加7")]
messages = app.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

