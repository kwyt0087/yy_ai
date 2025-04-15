# 导入 asyncio 模块，用于处理异步编程
#pip install langgraph
import asyncio

# 从 langgraph.checkpoint.sqlite.aio 模块中导入 AsyncSqliteSaver 类，它用于异步保存检查点到 SQLite 数据库
# pip install langgraph.checkpoint.sqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 从 langgraph.graph 模块中导入 StateGraph 类，它用于构建状态图
from langgraph.graph import StateGraph


async def main():
    # 创建一个 StateGraph 对象，节点的值类型为 int
    builder = StateGraph(int)

    # 添加一个名为 "add_one" 的节点，该节点的功能是将输入值加 1
    builder.add_node("add_one", lambda x: x + 1)

    # 设置 "add_one" 节点为状态图的入口点
    builder.set_entry_point("add_one")

    # 设置 "add_one" 节点为状态图的结束点
    builder.set_finish_point("add_one")

    # 使用异步上下文管理器创建一个 AsyncSqliteSaver 对象，并连接到名为 "checkpoints.db" 的 SQLite 数据库
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        # 编译状态图，并使用 memory 作为检查点保存器
        graph = builder.compile(checkpointer=memory)
        #print(graph.get_graph().draw_mermaid_png())
        # 创建一个异步调用状态图的协程，输入值为 1，并传入额外的配置参数
        result = await graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})

        # 打印结果
        print(result)


# 使用 asyncio.run 运行 main() 协程
asyncio.run(main())
