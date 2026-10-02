import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor


def query_and_print(query_engine, query):
    response = query_engine.query(query)
    answer = str(response)
    if not answer.strip() or answer.strip() == "Empty Response":
        raise RuntimeError("DeepSeek 未返回回答内容，请检查模型可用性和输出 token 限制。")
    print(f"回答: {answer}\n")


# 1. 配置模型
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_MODEL = "deepseek-flash"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError(
        "未读取到 DeepSeek API Key。请在项目根目录的 .env 中配置 "
        "DEEPSEEK_API_KEY，或在运行程序的终端中 export DEEPSEEK_API_KEY。"
        "如果密钥放在 ~/.bashrc，请使用已加载该文件的交互式 Bash 运行。"
    )

# 当前安装的 DeepSeek 适配器会丢弃 context_window，使用兼容适配器连接官方 API。
Settings.llm = OpenAILike(
    model=DEEPSEEK_MODEL,
    api_key=deepseek_api_key,
    api_base="https://api.deepseek.com",
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=32768,  # 为检索问答设置保守的上下文预算。
    max_tokens=4096,
    temperature=0.1,
    # 文档问答关闭思考模式，避免思考内容耗尽回答的 token 预算。
    additional_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
)
print("[1/5] 正在加载嵌入模型 BAAI/bge-small-en（首次运行需要下载）...", flush=True)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 2. 加载文档
print("[2/5] 正在读取 IPCC PDF 文档...", flush=True)
documents = SimpleDirectoryReader(
    input_files=[str(PROJECT_ROOT / "data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf")]
).load_data()
print(f"已读取 {len(documents)} 页。", flush=True)

# 3. 创建节点与构建索引
# 3.1 句子窗口索引
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
sentence_nodes = node_parser.get_nodes_from_documents(documents)
print(f"[3/5] 正在构建句子窗口索引：{len(sentence_nodes)} 个节点...", flush=True)
sentence_index = VectorStoreIndex(sentence_nodes, show_progress=True)

# 3.2 常规分块索引 (基准)
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
print(f"[4/5] 正在构建常规分块索引：{len(base_nodes)} 个节点...", flush=True)
base_index = VectorStoreIndex(base_nodes, show_progress=True)

# 4. 构建查询引擎
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 5. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"[5/5] 正在检索并调用 DeepSeek {DEEPSEEK_MODEL} 生成回答...", flush=True)
print(f"查询: {query}\n")

print("--- 句子窗口检索结果 ---")
query_and_print(sentence_query_engine, query)

print("--- 常规检索结果 ---")
query_and_print(base_query_engine, query)
