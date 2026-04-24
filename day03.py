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

# 定义工具
@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    # 用检索器搜索，返回最相关的文档片段列表
    docs_list = retriever.invoke(query)
    # 把每个文档片段的文本内容拼成一个字符串
    return "\n\n".join([doc.page_content for doc in docs_list])

# 工具
retriever_tool = retrieve_blog_posts

# 测试工具
retriever_tool.invoke({"query": "介绍一下凯文杜兰特"})
