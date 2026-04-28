# ### Day3 (4/23) 官方Agentic RAG教程
# - [ ] 精读 docs.langchain.com/oss/python/langgraph/agentic-rag
# - [ ] 理解CRAG: 检索→评分→路由
# - [ ] 逐行跟跑教程代码

# 导入环境变量
from dotenv import load_dotenv

load_dotenv()
import os
# 导入大模型，这里使用deepseek
from langchain_deepseek import ChatDeepSeek

# 定义大模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DeepSeek_API_KEY"),
    base_url=os.getenv("Base_URL"),
)

# 导入嵌入模型，这里使用智谱embedding-3
from langchain_community.embeddings import ZhipuAIEmbeddings

# 定义嵌入模型，这里使用智谱embedding-3
embedding = ZhipuAIEmbeddings(
    model = "embedding-3",
    api_key = os.getenv("zhipuai_api_key"),
)

# 1.预处理文件，这里使用网站内容
# "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
# "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
# "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",

# # 导入网页读取器，WebBaseLoader
# from langchain_community.document_loaders import WebBaseLoader

# # 地址
# urls = [
#     "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
#     "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
#     "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
# ]

# # 读取网页内容
# docs = [WebBaseLoader(url).load() for url in urls]

# # 打印读取的文档数
# print(f"读取的文档数: {len(docs)}")
# # 打印前1000行内容预览，
# # strip() 去除字符串首尾的空白字符（空格、换行、制表符）。
# print(docs[0][0].page_content.strip()[:1000])

# 读取本地文件
# 导入本地文件读取器
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import glob
# 定义docs字典，放入读取的文档
docs = []

# 读取目录下的文档
for pdf_path in glob.glob("./data/*.pdf"):
    docs.extend(PyPDFLoader(pdf_path).load())
for txt_path in glob.glob("./data/*.txt"):
    docs.extend(TextLoader(txt_path, encoding="utf-8").load())

# 打印读取的文档数
print(f"读取的文档数为：{len(docs)}")
# 打印读取文档的总字符数
print(f"读取的文档总字符数为：{len(docs[0].page_content)}")
# 打印前500字预览
print(f"读取的文档前500字符预览：{docs[0].page_content[:500]}")

# 拆分文档
# 导入文本分割器，RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 文档列表
# docs_list = [item for sublist in docs for item in sublist]

# 文本分割器，from_tiktoken_encoder()以token数来分割。
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50,
)

# 分割后的文档
doc_splits = text_splitter.split_documents(docs)
# 打印分割后的文档数量
print(f"分割后的文档数量: {len(doc_splits)}")
# 预览文档内容
print(doc_splits[0].page_content.strip())

# 导入向量存储库chromdb
from langchain_chroma import Chroma

# 创建向量存储
vectorstore = Chroma(
    collection_name="agentic_rag_collection",
    embedding_function=embedding,
    persist_directory="./chroma_crag_db",
)

# 清空旧数据，避免重复嵌入
vectorstore.reset_collection()
print("旧数据已清空")
# 将切分后的文档转化为向量，存入向量数据库
document_ids = vectorstore.add_documents(documents=doc_splits)
print(f"向量数据库已更新，新增文档ID: {document_ids}")

# 创建检索器
retriever = vectorstore.as_retriever()

# 包装retriever为工具
# 导入工具包
from langchain.tools import tool

# -----------------------------------------------
# 定义工具
# -----------------------------------------------

@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    # 用检索器搜索，返回最相关的文档片段列表
    docs_list = retriever.invoke(query)
    # 把每个文档片段的文本内容拼成一个字符串
    return "\n\n".join([doc.page_content for doc in docs_list])

# 工具
retriever_tool = retrieve_blog_posts

# # 测试工具
# retriever_tool.invoke({"query": "介绍一下凯文杜兰特"})

# 构建节点与边，
# ①构建一个节点。它会调用一个大型语言模型，
# 根据当前图态（消息列表）生成响应。根据输入消息，它会决定用检索工具检索，还是直接回复用户。
# 导入messagestate，记录图的消息列表.
# LangGraph 的消息状态类型，和 Day1 用的一样，存储对话消息列表

# ---------------------------------
# 定义完毕
# ---------------------------------


# ---------------------------------
# 定义节点，让 LLM 看到用户问题后自己决定是检索还是直接回答
# ---------------------------------
from langgraph.graph import MessagesState

# 定义回答模型
response_model = llm

# 定义节点函数
# 让 LLM 看到用户问题后自己决定是检索还是直接回答
def generate_query_or_response(state: MessagesState):
    """使模型基于当前的状态生成回答。对于所给出的问题，大模型会决定使用检索工具来检索还是直接
    恢复用户"""
    response = (
        response_model
        # .bind_tools([retriever_tool]) — 告诉 LLM "你可以调用这个检索工具"
        .bind_tools([retriever_tool])
        # 把对话历史发给 LLM
        .invoke(state["messages"])
    )
    # 把 LLM 的回复加入状态
    return {"messages": [response]}

# 测试随机输入
# input = {"messages":[{"role": "user", "content": "你好"}]}
# print(generate_query_or_response(input)["messages"][-1].pretty_print())

# 提出一个需要检索的问题
# input = {"messages":[{"role": "user", "content": "介绍一下NBA中的凯文杜兰特"}]}
# print(generate_query_or_response(input)["messages"][-1].pretty_print())


# ---------------------------------
# 定义完毕
# ---------------------------------


# # ---------------------------------
# 定义条件边
# ---------------------------------

# 添加一个条件边——以判断检索到的文档是否与问题相关。
# 我们将使用带有结构化输出模式的模型进行文档评分。
# 该函数会根据评分决策返回要访问的节点名称：

# 导入结构化输出模式，
# BaseModel — Pydantic 的基类，定义数据结构，LLM 必须按这个格式输出
# Field — 定义字段的元数据，比如描述、示例等，给字段加描述，告诉 LLM 这个字段该填什么
from pydantic import BaseModel, Field
# 导入类型, 限制值只能是 "yes" 或 "no"，不能随便写
from typing import Literal

# 打分提示词
GRADE_PROMPT = (
    """You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
)

# 定义数据结构
class GradeDocuments(BaseModel):
    """文档相关性评分"""
    
    binary_score: str = Field(
        description="文档是否与问题相关，如果相关‘yes’，否则‘no’"
    )

# 打分模型
grader_model = llm

# 定义条件边函数，
# 返回值决定走哪个分支返回 "generate_answer" → 文档相关，直接生成回答
# 返回 "rewrite_question" → 文档不相关，改写问题重新检索

def  grade_documents(
    state: MessagesState
) -> Literal["generate_answer", "rewrite_question"]:
    """判断文档是否与问题相关"""
    # 取第一条消息的内容，即用户的原始问题
    question = state["messages"][0].content
    # 取最后一条消息的内容，即检索工具返回的文档片段
    context = state["messages"][-1].content

    # 将问题与文档片段放入提示词
    prompt = GRADE_PROMPT.format(question=question, context=context)

    # 调用模型回答
    response = (
        grader_model
        .with_structured_output(GradeDocuments)
        .invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    
    # 打分
    score = response.binary_score

    # 根据评分决定走那个分支
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
# ---------------------------------------------
# 手动测试
# ---------------------------------------------
# 手动测试，逻辑是否正确，文档不相关
# 导入langchain的convert_to_message
# convert_to_messages() — 把字典列表转成 LangChain 的 Message 对象
# from langchain_core.messages import convert_to_messages

# input = {
#     "messages": convert_to_messages(
#         [
#             # 用户问题
#             {
#                 "role": "user",
#                 "content": "NBA是美国男子篮球联赛吗？",
#             },
#             # LLM回答
#             {
#                 "role": "assistant",
#                 "content": "",
#                 # 调用工具
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         # 模拟给检索工具 retrieve_blog_posts 的参数。
#                         "args": {"query": "美国男子篮球联赛"},
#                     }
#                 ],
#             },
#             # 检索结果故意写了 "meow"（猫叫声），和问题完全不相关
#             # 所以 grade_documents 应该返回 "rewrite_question"（不相关，需要改写问题）
#             {"role": "tool", "content": "meow", "tool_call_id": "1"},
#         ]
#     )
# }
# print(grade_documents(input))

# # 手动测试，文档相关
# input = {
#     "messages": convert_to_messages(
#         [
#             # 用户问题
#             {
#                 "role": "user",
#                 "content": "NBA是美国男子篮球联赛吗？",
#             },
#             # LLM回答
#             {
#                 "role": "assistant",
#                 "content": "",
#                 # 调用工具
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         # 模拟给检索工具 retrieve_blog_posts 的参数。
#                         "args": {"query": "美国男子篮球联赛"},
#                     }
#                 ],
#             },
#             # 检索结果故意写了“NBA是美国男子篮球联赛”，和问题相关
#             # 所以 grade_documents 应该返回 "rewrite_question"（相关，需要改写问题）
#             {
#                 "role": "tool", 
#                 "content": "NBA是美国男子篮球联赛", 
#                 "tool_call_id": "1"
#             },
#         ]
#     )
# }
# print(grade_documents(input))
# ---------------------------------------------
# 测试完毕
# ---------------------------------------------


# ---------------------------------
# 定义完毕
# ---------------------------------



# ---------------------------------------------
# 构建节点rewrite_question（不相关，需要改写问题） 
# generate_answer（相关，生成回答）
# ---------------------------------------------

# 构建rewrite_question节点

# 导入humanmessage
from langchain_core.messages import HumanMessage

# 重写提示词
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

# 定义改写问题节点函数，当 grade_documents 返回 "rewrite_question" 时执行
def rewrite_question(state: MessagesState):
    """改写用户原始问题"""
    # 提取消息
    messages = state["messages"]
    # 提取用户问题
    question = messages[0].content
    # 构建提示词，将用户问题放入提示词
    prompt = REWRITE_PROMPT.format(question=question)
    # 让 LLM 根据提示词改写问题（不绑定工具，纯文本生成）
    response = response_model.invoke([{"role": "user", "content": prompt}])
    # 把改写后的问题作为新的 HumanMessage 加入状态
    # 用 HumanMessage 而不是 AIMessage，因为改写后的问题要重新走检索流程，
    # 相当于"用户换了个问法"
    return {"messages": [HumanMessage(content=response.content)]}
    
# 构建generate_answer节点

# generate_answer节点提示词
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

# 定义generate_answer节点函数，当 grade_documents 返回 "generate_answer" 时执行
def generate_answer(state: MessagesState):
    """生成答案"""
    # 提取问题
    question = state["messages"][0].content
    # 取最后一条消息，即检索工具返回的相关文档内容
    context = state["messages"][-1].content
    # 构建提示词，将问题与文档内容放入提示词模板
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    # 调用LLM生成答案
    response = response_model.invoke([{"role": "user", "content": prompt}])
    # 返回生成的答案
    return {"messages": [response]}

# ---------------------------------------------
# 构建完毕
# ---------------------------------------------


# ---------------------------------------------
# 构建graph
# ---------------------------------------------

# 导入langgraph的StateGraph,START,END
from langgraph.graph import StateGraph, START, END
# 导入预构建的ToolNode，tools_condition
from langgraph.prebuilt import ToolNode, tools_condition

# 创建图，用 MessagesState 管理状态
Graph =  StateGraph(MessagesState)

# 添加节点
Graph.add_node(generate_query_or_response)
Graph.add_node("retrieve", ToolNode([retriever_tool]))
Graph.add_node(rewrite_question)
Graph.add_node(generate_answer)

# 添加边
Graph.add_edge(START, "generate_query_or_response")

# 添加条件边，决定是否检索
Graph.add_conditional_edges(
    "generate_query_or_response",
    # tools_condition — LangGraph 预构建的条件函数，检查 tool_calls 是否存在
    tools_condition,
    {
        "tools": "retrieve",# 调用了工具 → 去检索
        END: END,# 没调用工具 → 直接结束（LLM已直接回答）
    },
)

# 检索完成后，grade_documents 决定走哪个分支
Graph.add_conditional_edges(
    "retrieve",
    # 评分函数，返回 "generate_answer" 或 "rewrite_question"
    # 返回 "generate_answer" → 去生成回答
    # 返回 "rewrite_question" → 去改写问题
    grade_documents,
)
# 添加边
# 生成回答后结束
Graph.add_edge("generate_answer", END)
# 改写问题后回到起点，用新问题重新走整个流程（检索→评分→...）
Graph.add_edge("rewrite_question", "generate_query_or_response")

# 编译图
graph = Graph.compile()

# 可视化图表
# from IPython.display import Image, display

# display(Image(graph.get_graph().draw_mermaid_png()))
with open("crag_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

# 运行问题，测试图
# 流式执行图，每执行完一个节点就返回一个 chunk
for chunk in graph.stream(
    {
        "messages":[ # 传入初始状态：用户问题
            {
                "role": "user",
                "content": "NBA总冠军最多的球队是那个球队？"
            }
        ]
    }
):
    # chunk 是一个字典 {"节点名": 状态更新}
    # node — 刚执行完的节点名
    # update — 该节点返回的状态更新
    for node, update in chunk.items(): 
        # 打印节点更新情况
        print(f"更新节点 {node}")
        # 格式化打印消息内容（区分 Human/AI/Tool Message）
        update["messages"][-1].pretty_print()
        print("\n\n")
        