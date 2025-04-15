import asyncio
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# 定义一个工具函数，用于获取天气信息
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


# 创建一个包含工具函数的列表
tools = [get_weather]

# 初始化一个OpenAI的聊天模型，使用gpt-4o模型，温度设为0（生成结果更确定）
model = ChatOpenAI(model_name="gpt-4o", temperature=0)


# 定义一个异步主函数
async def main():
    # 创建一个反应式代理，使用聊天模型和工具
    graph = create_react_agent(model, tools)
    # 定义输入消息，包含一个人类的消息询问sf的天气
    inputs = {"messages": [("human", "what's the weather in sf")]}
    # 异步迭代获取代理的更新流
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        # 遍历每个更新块中的节点和值
        for node, values in chunk.items():
            # 打印接收到更新的节点名称
            print(f"Receiving update from node: '{node}'")
            # 打印节点的值
            print(values)
            print()


# 运行异步函数
asyncio.run(main())
