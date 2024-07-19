from utils import DBUtils
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.utilities import SQLDatabase



class TransformAgent():
    
    def __init__(self, uri:str='xxxx', model:str="gpt-3.5-turbo-0125") -> None:
        dbb = SQLDatabase.from_uri(uri)

        DBUtils.initialize(dbb)

        system_prompt = '''你是一名有经验的数据分析师，你现在需要教实习生如何针对用户需求取数，请一步一步思考写出你的教学。
        # 要求
        - 先写出你对用户需求的理解，可以将用户需求改写成更有利于指导取数的指令。
        - 通过提供的工具查看数据库中的表结构，在了解需要的表结构信息之前不要回答。
        - 不需要写出sql
        # 输出格式
        请以markdown格式输出，例子如下：
        ```
        # 用户需求理解：
        [你的需求理解]
        # 取数思路：
        - 第一步：...
        - 第二步：...
        - ...
        ```
        ！注意：回答不要超过200字/词，不要超过十句话。
        '''

        llm = ChatOpenAI(model=model, temperature=0)

        prompt =ChatPromptTemplate.from_messages(
        [
        ('system',system_prompt),
        # MessagesPlaceholder(variable_name='history'),
        ('user','{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
        ]
        )

        print(DBUtils.get_table_names)
        agent = (
            # {'dialect': lambda x:x['db'].dialect, 'ds_info': lambda x:x['db'].get_table_info(), 'input': lambda x:x['input']}
        {'input': lambda x:x['input'],
        'agent_scratchpad': lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
        }
        |
        prompt#.partial(format_instructions=parser.get_format_instructions())
        |
        llm.bind_tools([DBUtils.get_table_names, DBUtils.get_column_info])#.assign(hello=achain)
        |
        OpenAIToolsAgentOutputParser()
        # OpenAIToolsAgentOutputParser()
        )

        self.agent_executor = AgentExecutor(agent=agent, tools=[DBUtils.get_table_names, DBUtils.get_column_info], verbose=True, stream_runnable=False)
        # agent_executor = NewAgentExecutor(agent=agent, tools=[get_table_names, get_column_info, legitimacy_check], verbose=True, stream_runnable=False)
        # chain = agent_executor| (lambda x: parser.invoke(x['output']).sql)

    def invoke(self, input:str):
        answer = self.agent_executor.invoke({'input':input})
        return answer