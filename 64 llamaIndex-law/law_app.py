# -*- coding: utf-8 -*-
# 这行指定了文件的编码格式为UTF-8，确保中文字符能够正确显示
# 导入必要的库
# json库用于处理JSON格式数据，例如从文件中读取法律条文
import json
# time库用于计时，比如记录索引加载时间
import time
# pathlib库提供了面向对象的文件系统路径操作，比如查找所有JSON文件
from pathlib import Path
# typing库提供类型提示功能，帮助IDE提供更好的代码补全和错误检查
from typing import List, Dict

# chromadb是一个向量数据库，用于存储和检索文本的向量表示
import chromadb
# 以下是llama_index库的各种组件，用于构建检索增强生成(RAG)系统
# VectorStoreIndex是核心索引类，用于管理文档的向量表示
from llama_index.core import VectorStoreIndex, StorageContext, Settings
# TextNode是文档的基本单位，每个法律条文会被转换为一个TextNode
from llama_index.core.schema import TextNode
# HuggingFaceLLM是对接HuggingFace模型的接口，用于生成回答
from llama_index.llms.huggingface import HuggingFaceLLM
# HuggingFaceEmbedding用于将文本转换为向量表示
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# ChromaVectorStore是llama_index与chromadb的连接器
from llama_index.vector_stores.chroma import ChromaVectorStore
# PromptTemplate用于定义提示模板，指导LLM如何回答问题
from llama_index.core import PromptTemplate

# 定义问答模板，这是给大模型的指令和上下文格式
# 模板包含系统指令、相关法律条文和用户问题
# {context_str}和{query_str}是占位符，会在运行时被替换为实际内容
QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个专业的法律助手，请严格根据以下法律条文回答问题：\n"
    "相关法律条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# 创建提示模板对象，后续会用于查询引擎
response_template = PromptTemplate(QA_TEMPLATE)

# ================== 配置区 ==================
# 定义Config类，集中管理所有配置参数，方便修改和维护
class Config:
    # 嵌入模型路径，用于将文本转换为向量
    EMBED_MODEL_PATH = r"D:\software\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620"
    # 大语言模型路径，用于生成回答
    LLM_MODEL_PATH = r"D:\software\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B\snapshots\6393b7559e403fd1d80bfead361586fd6f630a4d"
    
    # 数据目录，存放法律条文的JSON文件
    DATA_DIR = "./data"
    # 向量数据库目录，存放持久化的向量数据
    VECTOR_DB_DIR = "./chroma_db"
    # 存储目录，用于保存索引和其他持久化数据
    PERSIST_DIR = "./storage"
    
    # 集合名称，在向量数据库中标识这组法律数据
    COLLECTION_NAME = "chinese_labor_laws"
    # 检索时返回的最相关条目数量
    TOP_K = 3

# ================== 初始化模型 ==================
# 初始化并验证嵌入模型和大语言模型
def init_models():
    """初始化模型并验证"""
    # 初始化嵌入模型，用于将文本转换为向量
    # 例如将"劳动合同"转换为一个512维的向量
    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL_PATH,
        device = 'cuda' if hasattr(Settings, 'device') else 'cpu'  # 如果可用则使用GPU加速
    )
    
    # 初始化大语言模型，用于生成回答
    # 例如根据检索到的法律条文，生成对"试用期最长多久"的专业回答
    llm = HuggingFaceLLM(
        model_name=Config.LLM_MODEL_PATH,
        tokenizer_name=Config.LLM_MODEL_PATH,
        device_map= "auto",  # 自动选择设备
        tokenizer_kwargs={"trust_remote_code": True},  # 信任模型的远程代码
        generate_kwargs={"temperature": 0.3}  # 温度参数控制生成的随机性，越低越确定性
    )
    
    # 将模型设置为全局默认模型
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # 验证嵌入模型是否正常工作，测试其输出维度
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")
    
    return embed_model, llm

# ================== 数据处理 ==================
# 加载并验证JSON格式的法律文件
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    # 查找目录中所有JSON文件
    # 例如找到 labor_law.json, labor_contract.json 等
    json_files = list(Path(data_dir).glob("*.json"))
    # 确保至少找到一个JSON文件，否则抛出异常
    assert json_files, f"未找到JSON文件于 {data_dir}"
    
    # 存储所有加载的数据
    all_data = []
    # 遍历每个JSON文件
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                # 加载JSON数据
                data = json.load(f)
                # 验证数据结构：根元素必须是列表
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                # 验证列表中的每个元素必须是字典
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    # 验证字典中的每个值必须是字符串
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                # 将数据添加到总列表中，同时记录来源文件
                all_data.extend({
                    "content": item,
                    "metadata": {"source": json_file.name}
                } for item in data)
            except Exception as e:
                # 如果处理过程中出现任何错误，抛出异常
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")
    
    # 打印加载成功的条目数量
    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data

# 将原始数据转换为TextNode对象，每个法律条文一个节点
def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """添加ID稳定性保障"""
    nodes = []
    # 遍历每个数据条目
    for entry in raw_data:
        # 获取法律条文字典和来源文件
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]
        
        # 遍历字典中的每个条目（法律条文）
        for full_title, content in law_dict.items():
            # 生成稳定ID，确保每次运行生成相同的ID
            # 例如："labor_law.json::中华人民共和国劳动法 第一条"
            node_id = f"{source_file}::{full_title}"
            
            # 分割标题，提取法律名称和条款号
            # 例如："中华人民共和国劳动法 第一条" -> "中华人民共和国劳动法", "第一条"
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"
            
            # 创建TextNode对象，包含条文内容和元数据
            node = TextNode(
                text=content,  # 条文内容
                id_=node_id,   # 稳定ID
                metadata={     # 元数据，用于后续检索和展示
                    "law_name": law_name,         # 法律名称
                    "article": article,           # 条款号
                    "full_title": full_title,     # 完整标题
                    "source_file": source_file,   # 来源文件
                    "content_type": "legal_article"  # 内容类型
                }
            )
            nodes.append(node)
    
    # 打印生成的节点数量和第一个节点的ID示例
    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes

# ================== 向量存储 ==================
# 初始化向量存储，创建或加载索引
def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    # 创建持久化的ChromaDB客户端，数据将保存在指定目录
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    # 获取或创建集合，用于存储法律条文的向量表示
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度计算向量距离
    )

    # 创建存储上下文，连接到ChromaDB
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    # 判断是否需要新建索引：如果集合为空且提供了节点，则创建新索引
    if chroma_collection.count() == 0 and nodes is not None:
        print(f"创建新索引（{len(nodes)}个节点）...")
        
        # 将节点添加到文档存储中
        storage_context.docstore.add_documents(nodes)  
        
        # 创建向量索引
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True  # 显示进度条
        )
        # 双重持久化保障，确保数据不会丢失
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)  # <-- 新增
    else:
        # 如果集合不为空，则加载已有索引
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        # 从向量存储创建索引
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # 安全验证，确保数据正确加载
    print("\n存储验证结果：")
    # 检查文档存储中的记录数
    doc_count = len(storage_context.docstore.docs)
    print(f"DocStore记录数：{doc_count}")
    
    # 如果有记录，打印一个示例节点ID
    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"示例节点ID：{sample_key}")
    else:
        # 如果没有记录，打印警告
        print("警告：文档存储为空，请检查节点添加逻辑！")
    
    
    return index

# ================== 主程序 ==================
# 主函数，程序入口
def main():
    # 初始化模型
    embed_model, llm = init_models()
    
    # 仅当需要更新数据时执行
    # 如果向量数据库目录不存在，则需要从头创建
    if not Path(Config.VECTOR_DB_DIR).exists():
        print("\n初始化数据...")
        # 加载原始数据
        raw_data = load_and_validate_json_files(Config.DATA_DIR)
        # 创建文本节点
        nodes = create_nodes(raw_data)
    else:
        # 如果已有数据，则不需要重新加载
        nodes = None
    
    # 初始化向量存储
    print("\n初始化向量存储...")
    # 记录开始时间，用于计算加载耗时
    start_time = time.time()
    # 创建或加载索引
    index = init_vector_store(nodes)
    # 打印加载耗时
    print(f"索引加载耗时：{time.time()-start_time:.2f}s")
    
    # 创建查询引擎，用于检索和回答问题
    query_engine = index.as_query_engine(
        similarity_top_k=Config.TOP_K,  # 检索最相关的TOP_K个条目
        text_qa_template=response_template,  # 使用前面定义的提示模板
        verbose=True  # 显示详细日志
    )
    
    # 交互式问答循环
    while True:
        # 获取用户输入的问题
        question = input("\n请输入劳动法相关问题（输入q退出）: ")
        # 如果用户输入q，则退出循环
        if question.lower() == 'q':
            break
        
        # 执行查询，获取回答
        response = query_engine.query(question)
        
        # 显示回答结果
        print(f"\n智能助手回答：\n{response.response}")
        # 显示支持依据（检索到的相关法律条文）
        print("\n支持依据：")
        # 遍历每个来源节点
        for idx, node in enumerate(response.source_nodes, 1):
            # 获取节点元数据
            meta = node.metadata
            # 打印条目编号和标题
            print(f"\n[{idx}] {meta['full_title']}")
            # 打印来源文件
            print(f"  来源文件：{meta['source_file']}")
            # 打印法律名称
            print(f"  法律名称：{meta['law_name']}")
            # 打印条款内容（前100个字符）
            print(f"  条款内容：{node.text[:100]}...")
            # 打印相关度得分
            print(f"  相关度得分：{node.score:.4f}")

# 程序入口点，当直接运行此脚本时执行main函数
if __name__ == "__main__":
    main()