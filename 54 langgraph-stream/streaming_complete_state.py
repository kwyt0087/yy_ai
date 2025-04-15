# 导入异步IO模块
import asyncio
# 从类型提示模块中导入Literal
from typing import Literal
# 导入工具装饰器
from langchain_core.tools import tool
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入创建react代理的函数
from langgraph.prebuilt import create_react_agent


# 定义一个工具函数
@tool
# 定义一个获取天气的函数，参数city只能是"nyc"或"sf"
def get_weather(city: Literal["nyc", "sf"]):
    # 函数的文档字符串，描述函数用途
    """Use this to get weather information."""
    # 如果城市是纽约
    if city == "nyc":
        # 返回纽约的天气信息
        return "It might be cloudy in nyc"
    # 如果城市是旧金山
    elif city == "sf":
        # 返回旧金山的天气信息
        return "It's always sunny in sf"
    # 如果城市不是nyc或sf
    else:
        # 抛出一个断言错误
        raise AssertionError("Unknown city")


# 将get_weather函数放入工具列表
tools = [get_weather]

# 创建一个OpenAI聊天模型实例，使用gpt-4o模型，温度设为0
model = ChatOpenAI(model_name="gpt-4o", temperature=0)
# 使用模型和工具创建一个react代理
graph = create_react_agent(model, tools)


# 定义一个异步主函数
async def main():
    # 定义输入消息，询问旧金山的天气
    inputs = {"messages": [("human", "what's the weather in sf")]}
    # 异步迭代代理的输出流，模式为"values"
    async for chunk in graph.astream(inputs, stream_mode="values"):
        # 打印最后一条消息的格式化内容
        chunk["messages"][-1].pretty_print()

    # 再次定义输入消息，询问旧金山的天气
    inputs = {"messages": [("human", "what's the weather in sf")]}
    # 异步迭代代理的输出流，模式为"values"
    async for chunk in graph.astream(inputs, stream_mode="values"):
        # 将最后一个chunk赋值给final_result
        final_result = chunk

    # 打印final_result
    print(final_result)
    # 打印final_result中最后一条消息的格式化内容
    print(final_result["messages"][-1].pretty_print())


# 运行异步函数
asyncio.run(main())
