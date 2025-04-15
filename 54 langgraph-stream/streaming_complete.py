import asyncio
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

model = ChatOpenAI(model_name="gpt-4o", temperature=0)
graph = create_react_agent(model, tools)


async def main():
    # 使用 values 模式
    # inputs = {"messages": [("human", "what's the weather in sf")]}
    # print("Using values mode:")
    # async for chunk in graph.astream(inputs, stream_mode="values"):
    #     print(chunk)

    # 使用 updates 模式
    inputs = {"messages": [("human", "what's the weather in sf")]}
    print("Using updates mode:")
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        print(chunk)


# 运行异步函数
asyncio.run(main())
