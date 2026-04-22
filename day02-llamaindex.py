# ### Day2 LlamaIndex实现RAG
# 对比LangChain版本(day02.py)，LlamaIndex把分块、嵌入、存储、检索都封装好了
# 核心只需：加载文档 → 构建索引 → 查询

# =============================================================================
# LlamaIndex实现RAG
# =============================================================================
# 导入环境变量
from dotenv import load_dotenv
load_dotenv()
import os
# 导入大模型这里使用deepseek
from llama_index.llms.deepseek import DeepSeek

# 定义大模型
llm = DeepSeek(
    model = "deepseek-chat",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("Base_URL"),
)

# 导入嵌入模型，这里使用智谱嵌入模型，embedding-3
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

# 定义嵌入模型
embedding = ZhipuAIEmbedding(
    model="embedding-3",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
)

# 导入文档加载器与向量索引
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

# 读取文档
documents = SimpleDirectoryReader("./data").load_data()
# 打印读取了多少文档，分别是什么
print(f"读取了{len(documents)}个文档")
for i, doc in enumerate(documents):
    print(f"文档{i}: {doc.metadata.get('file_name', 'unknown')}-{len(doc.text)}字符")

# 切分文档，构建索引
# 设置全局llm和embedding llm
Settings.llm = llm
Settings.embed_model = embedding

# 构建索引（自动分块+嵌入+存储）
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,# 显示进度条
)

# 创建查询引擎，as_query_engine() 创建查询引擎，
# 自动完成：检索 → 拼接上下文 → LLM生成回答。
# 对比LangChain需要手动：similarity_search → 拼接 → model.invoke
query_engine = index.as_query_engine()

# 定义问题
query = "NBA有哪些著名的球队？用中文回答"
# 执行查询
result = query_engine.query(query)
print(f"\n问题: {query}")
print(f"回答: {result}")

# ---------------------------------------------------------------------------
# 持久化存储（可选）
# ---------------------------------------------------------------------------
# 默认存在内存中，程序结束就没了。可以持久化到磁盘：
# index.storage_context.persist(persist_dir="./llamaindex_storage")
# 下次加载时：
# from llama_index.core import StorageContext, load_index_from_storage
# storage_context = StorageContext.from_defaults(persist_dir="./llamaindex_storage")
# index = load_index_from_storage(storage_context)

# ---------------------------------------------------------------------------
# 高级用法：自定义分块（可选）
# ---------------------------------------------------------------------------
# 如果想控制分块策略（类似LangChain的RecursiveCharacterTextSplitter）：
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import VectorStoreIndex
#
# text_splitter = SentenceSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )
# index = VectorStoreIndex.from_documents(
#     documents,
#     transformations=[text_splitter],
# )

# ---------------------------------------------------------------------------
# 使用chromadb持久化存储
# ---------------------------------------------------------------------------

# import chromadb
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
# from llama_index.vector_stores.chroma import ChromaVectorStore

# # 创建Chroma客户端
# chroma_client = chromadb.PersistentClient(path="./chroma_llamaindex_db")
# chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("llamaindex"))
# storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)

# # 构建索引时指定storage_context
# index = VectorStoreIndex.from_documents(
#     documents,
#     storage_context=storage_context,
# )
