from langchain_community.utilities.sql_database import SQLDatabase
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_openai import ChatOpenAI
import json
from langchain_core.agents import AgentActionMessageLog, AgentFinish, AgentStep
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import (
#     format_to_openai_tool_messages,
# )
import dotenv
import os
import json
from typing import Dict, List, Sequence, Tuple
import copy
from langchain_core.agents import AgentAction
from langchain.agents.output_parsers.openai_tools import parse_ai_message_to_openai_tool_action 
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
import sqlparse
from sqlparse.sql import Statement
from typing import Union
from utils import DBUtils
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from langsmith import traceable


dotenv.load_dotenv()




def _create_tool_message(
    agent_action: AgentActionMessageLog, observation: str
) -> ToolMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return ToolMessage(
        tool_call_id=agent_action.tool_call_id,
        content=content,
        additional_kwargs={"name": agent_action.tool},
    )


def format_to_openai_tool_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    Returns:
        list of messages to send to the LLM for the next prediction

    """
    messages = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, AgentActionMessageLog):
            new_messages = list(agent_action.message_log) + [
                _create_tool_message(parse_ai_message_to_openai_tool_action(agent_action.message_log[0])[0], observation)
            ]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
    return messages


class NL2SQL:
    embedding_model = OpenAIEmbeddings()
    # 3. 如果您需要更多信息，可以使用提供的工具。如果问题无法解决，您可以向用户询问更多信息。
    @classmethod
    def initialize_agent(cls, database_uri:str, vectorbase_uri=None, model:str='gpt-4-turbo-preview'):
        system_prompt = '''你是一名经验丰富的数据库管理员，请根据数据源信息来编写SQL语句回答用户问题.
                约束:
                    1. 请根据用户问题理解用户意图。如果意图不明，可以向用户询问更多信息，或者给用户提供几条提问建议。以下是一些可能相关的表信息和知识点
                        ```
                        ```
                    2. 你的首选方案是参考过往的案例来辅助你编写SQL，以下是一些例子，如果你认为过往案例中有与用户问题类似的可以直接稍作改动生成sql，不需要再调用工具：
                            ```
                            {examples}
                            ```
                    3. 调用查看数据库结构的工具会极大增加开支，如果没有必要的理由请不要调用相关工具。
                    4. 创建一个语法正确的 {dialect}sql，如果不需要sql，则直接回答用户问题。
                    5. 使用submits_sql工具来提交生成的SQL
                请一步步思考
                处理接下来用户输入的问题
                '''
        
        db = SQLDatabase.from_uri(database_uri)

        NL2SQL.db_name = database_uri[database_uri.rindex('/')+1:]
        NL2SQL.question_to_source = Chroma(collection_name='quesiton_to_source', embedding_function=NL2SQL.embedding_model, persist_directory='./rag_test')
        NL2SQL.query_to_thoughts = Chroma(collection_name= 'query_to_thoughts', embedding_function=NL2SQL.embedding_model, persist_directory='./rag_test')


        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        DBUtils.initialize(db)

        # NL2SQL.vectorbase = Chroma(vectorbase_uri)
        # NL2SQL.vector_db = 
        # NL2SQL.vector_db1 = Chroma()

        prompt =ChatPromptTemplate.from_messages(
        [
        ('system',system_prompt),
        # MessagesPlaceholder(variable_name='history'),
        ('user','{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')
        ]
        )

        
        agent = (
            # {'dialect': lambda x:x['db'].dialect, 'ds_info': lambda x:x['db'].get_table_info(), 'input': lambda x:x['input']}
        {'input': lambda x:x['input'], 
        'agent_scratchpad': lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        'examples': lambda x:x['examples']
        }
        |
        prompt.partial(dialect=db.dialect)
        |
        llm.bind_tools([DBUtils.get_table_names, DBUtils.get_column_info, DBUtils.submit_sql])#.assign(hello=achain)
        |
        NL2SQL.parse
        # OpenAIToolsAgentOutputParser()
        )

        agent_executor = AgentExecutor(agent=agent, tools=[DBUtils.get_table_names, DBUtils.get_column_info, DBUtils.submit_sql], verbose=True, stream_runnable=False)
        
        chain = RunnableLambda(NL2SQL.retriever) | agent_executor

        return chain     
    
    @classmethod
    def switch_embedding_model(cls, model_name_or_path):
        #    if model_name_or_path:
        #       pass
       NL2SQL.embedding_model = model_name_or_path
       

    @staticmethod
    def retriever(input:str):
        # vector_db = Chroma(embedding_function=NL2SQL.embedding_model, persist_directory='./approach3')
        # vector_db1 = Chroma(embedding_function=NL2SQL.embedding_model, persist_directory='./steps_and_sqls')

        targets = [x[0] for x in 
                    NL2SQL.question_to_source.similarity_search_with_relevance_scores(input)
                    if x[0].metadata['database']==NL2SQL.db_name
        ]
        # question = targets[0].page_content

        column_names = []
        table_names = []
        for i in targets:
            source = json.loads(i.metadata['source'])
            column_names.extend([x for x in source['Columns'] if x not in column_names])
            table_names.extend([x for x in source['Tables'].split(',') if x not in table_names])

        query = f"{input},可能的表：{table_names}，可能的列{column_names}"

        res = [x[0] for x in 
                    NL2SQL.query_to_thoughts.similarity_search_with_relevance_scores(query,k=1)
                    if x[0].metadata['database']==NL2SQL.db_name
        ]
        return {'input':input,'examples':[{'Thinkings':x.page_content,'sql':x.metadata['sql']} for x in res]}
    

    @staticmethod
    def parse(output):
        # If no function was invoked, return to user
        if "tool_calls" not in output.additional_kwargs:
            return AgentFinish(return_values={"output": output.content}, log=output.content)

        # Parse out the function call
        tool_calls = output.additional_kwargs["tool_calls"]

        logs = []
        for call in tool_calls:
            name = call['function']['name']
            inputs = json.loads(call['function']["arguments"])
            if name == 'submit_sql':
                sql = inputs['sql']
                if DBUtils.is_select_statement(sql):
                    try:
                        DBUtils.db.run(sql)
                        # return f"The sql '{sql}' is legitimate and error free, you can submit it."
                        return AgentFinish(return_values=inputs, log=str(call))   
                    except Exception as e:
                        # return f"An error occured when running the sql {e}"
                        return AgentActionMessageLog(tool=name, tool_input=inputs, 
                                            log="", message_log=[output])        
                else:
                    return AgentActionMessageLog(tool=name, tool_input=inputs, 
                                        log="", message_log=[output])
            else:    
                output_ = copy.deepcopy(output)
                output_.additional_kwargs["tool_calls"] = [call]
                logs.append(AgentActionMessageLog(tool=name, tool_input=inputs, 
                                        log="", message_log=[output_]))
        
        # return AgentActionMessageLog(tool=logs[0][0], tool_input=logs[0][1], 
        #                              log="", message_log=[output])
        return logs


@traceable(run_type="llm",
    name="NL2SQLwithRAG",
    tags=["RAG1.0"],
    metadata={"database": "test_yishion"})
def main(input:str):
    chain = NL2SQL.initialize_agent('') #填入数据库连接信息
    return chain.invoke(input)


# # achain =  RunnableLambda(lambda x:print(x))
if __name__ == "__main__":
   
   main('以纯电商2024-04-22到2024-04-23的实际支付金额的同比增长率')
#    chain = NL2SQL.initialize_agent()
   

#   # agent_executor = NewAgentExecutor(agent=agent, tools=[get_table_names, get_column_info, legitimacy_check], verbose=True, stream_runnable=False)
#   # chain = agent_executor| (lambda x: parser.invoke(x['output']).sql)

#   answer = agent_executor.invoke({'db':db,'input':'List the top5 best-selling cars'})
