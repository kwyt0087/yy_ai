import asyncio
from langchain_core.messages import HumanMessage

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.types import StreamWriter

async def my_node(state: MessagesState, config: RunnableConfig):
    chunks = [
        "Four",
        "score",
        "and",
        "seven",
        "years",
        "ago",
        "our",
        "fathers",
        "...",
    ]
    for chunk in chunks:
        await adispatch_custom_event(
            "my_custom_event",
            {"chunk": chunk},
            config=config,  # <-- 传播配置
        )

    return {"messages": [AIMessage(content=" ".join(chunks))]}


# 定义一个新图表。
workflow = StateGraph(MessagesState)

workflow.add_node("model", my_node)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

app = workflow.compile()


# 定义一个异步主函数
async def main():
    inputs = [HumanMessage(content="What are you thinking about?")]
    async for event in app.astream_events({"messages": inputs}, version="v2"):
        tags = event.get("tags", [])
        if event["event"] == "on_custom_event" and event["name"] == "my_custom_event":
            data = event["data"]
            if data:
                print(data["chunk"], end="、", flush=True)


# 定义一个异步主函数
asyncio.run(main())