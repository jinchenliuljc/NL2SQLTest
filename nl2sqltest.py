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



dotenv.load_dotenv()
print(os.environ['LANGCHAIN_API_KEY'])

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# class RunPythonCodeInput(BaseModel):
#     code: str = Field(description="Python code to run", )


# class GetTableNames(BaseTool):
#     name = "get_table_names"
#     description = "Get all the table names and (possibly) along with each a brief description"
#     args_schema: Type[BaseModel] = RunPythonCodeInput

#     def _run(self):

class TableName(BaseModel):
    table_name: str = Field(description="the exact name of the table you want to explore.")



# class NewAgentExecutor(AgentExecutor):

#     def __init__(
#         self,
#         agent,
#         tools: Sequence[BaseTool],
#         verbose: bool = True,
#         stream_runnable: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             agent=agent,
#             tools=tools,
#             verbose=verbose,
#             stream_runnable=stream_runnable,
#             **kwargs,
#         )
    
#     def _iter_next_step(self, name_to_tool_map: Dict[str, BaseTool], 
#                         color_mapping: Dict[str, str], 
#                         inputs: Dict[str, str], 
#                         intermediate_steps, 
#                         run_manager) :
#         try:
#             intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

#             # Call the LLM to see what to do.
#             output = self.agent.plan(
#                 intermediate_steps,
#                 callbacks=run_manager.get_child() if run_manager else None,
#                 **inputs,
#             )
#         except OutputParserException as e:
#             if isinstance(self.handle_parsing_errors, bool):
#                 raise_error = not self.handle_parsing_errors
#             else:
#                 raise_error = False
#             if raise_error:
#                 raise ValueError(
#                     "An output parsing error occurred. "
#                     "In order to pass this error back to the agent and have it try "
#                     "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
#                     f"This is the error: {str(e)}"
#                 )
#             text = str(e)
#             if isinstance(self.handle_parsing_errors, bool):
#                 if e.send_to_llm:
#                     observation = str(e.observation)
#                     text = str(e.llm_output)
#                 else:
#                     observation = "Invalid or incomplete response"
#             elif isinstance(self.handle_parsing_errors, str):
#                 observation = self.handle_parsing_errors
#             elif callable(self.handle_parsing_errors):
#                 observation = self.handle_parsing_errors(e)
#             else:
#                 raise ValueError("Got unexpected type of `handle_parsing_errors`")
#             output = AgentAction("_Exception", observation, text)
#             if run_manager:
#                 run_manager.on_agent_action(output, color="green")
#             tool_run_kwargs = self.agent.tool_run_logging_kwargs()
#             observation = ExceptionTool().run(
#                 output.tool_input,
#                 verbose=self.verbose,
#                 color=None,
#                 callbacks=run_manager.get_child() if run_manager else None,
#                 **tool_run_kwargs,
#             )
#             yield AgentStep(action=output, observation=observation)
#             return

#         # If the tool chosen is the finishing tool, then we end and return.
#         if isinstance(output, AgentFinish):
#             yield output
#             return

#         actions: List[AgentAction]
#         if isinstance(output, AgentAction):
#             actions = [output]
#         else:
#             actions = output
#         for agent_action in actions:
#             yield agent_action
#         for agent_action in actions:
#             yield self._perform_agent_action(
#                 name_to_tool_map, color_mapping, agent_action, run_manager
#             )



@tool
def get_table_names():
    '''Get all the table names and (sometimes) along with each a brief description'''
    return db.run("SELECT name FROM sqlite_master WHERE type = 'table';")

@tool(args_schema=TableName)
def get_column_info(table_name):
    """Get the SQL that defines all the columns of the table, including type, references, etc."""
    return db.run(f"SELECT sql FROM sqlite_master WHERE name = '{table_name}'")

@tool
def submit_sql(sql:str):
    '''Submit the generated sql'''
    # print('sql:',sql)
    if is_select_statement(sql):
        try:
            db.run(sql)
            return f"The sql '{sql}' is legitimate and error free, you can submit it."
            # return AgentFinish(return_values=sql, log='')
        except Exception as e:
            return f"An error occured when running the sql {e}"        
    else:
        return 'You can only write sql that is SELECT statement'
    # return "sql submitted!"


# def get_template



def is_select_statement(sql):
    parsed = sqlparse.parse(sql)
    if not parsed:
        return False
    statement = parsed[0]  # 取第一个语句
    if isinstance(statement, Statement) and statement.get_type() == 'SELECT':
        return True
    return False

# @tool
# def legitimacy_check(sql:str):
#     '''Check the correctness and legitimacy of the generated sql'''
#     if is_select_statement(sql):
#         try:
#             db.run(sql)
#             # return f"The sql '{sql}' is legitimate and error free, you can submit it."
#             return AgentFinish(return_values=sql, log='')
#         except Exception as e:
#             return f"An error occured when running the sql {e}"        
#     else:
#         return 'You can only write sql that is SELECT statement'



# llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)



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
          if is_select_statement(sql):
            try:
              db.run(sql)
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





system_prompt = '''你是一名经验丰富的数据库管理员，请根据数据源信息来编写SQL语句回答用户问题.
        约束:
            1. 请根据用户问题理解用户意图。如果意图不明，可以向用户询问更多信息，或者给用户提供几条提问建议。
            2. 如果您需要更多信息，可以使用提供的工具。如果问题无法解决，您可以向用户询问更多信息。
            3. 创建一个语法正确的 {dialect}sql，如果不需要sql，则直接回答用户问题。
            4. 请不要输出任何内容，使用submits_sql工具来提交生成的SQL
        请一步步思考
        处理接下来用户输入的问题

'''



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



# achain =  RunnableLambda(lambda x:print(x))
if __name__ == "__main__":
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
  {'dialect': lambda x:x['db'].dialect, 'input': lambda x:x['input'], 
  'agent_scratchpad': lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
  }
  |
  prompt#.partial(format_instructions=parser.get_format_instructions())
  |
  llm.bind_tools([get_table_names, get_column_info, submit_sql])#.assign(hello=achain)
  |
  parse
  # OpenAIToolsAgentOutputParser()
  )

  agent_executor = AgentExecutor(agent=agent, tools=[get_table_names, get_column_info, submit_sql], verbose=True, stream_runnable=False)
  # agent_executor = NewAgentExecutor(agent=agent, tools=[get_table_names, get_column_info, legitimacy_check], verbose=True, stream_runnable=False)
  # chain = agent_executor| (lambda x: parser.invoke(x['output']).sql)

  answer = agent_executor.invoke({'db':db,'input':'List the top5 best-selling cars'})


  print(answer)



#标准的intermediate_step格式
'''[
      {
        "tool": "run_python_code",
        "tool_input": {
          "code": "import pandas as pd\n\ndf = pd.read_csv('/home/user/data_analysis10063527483056799834.csv')\nprint(df.head())"
        },
        "log": "\nInvoking: `run_python_code` with `{'code': \"import pandas as pd\\n\\ndf = pd.read_csv('/home/user/data_analysis10063527483056799834.csv')\\nprint(df.head())\"}`\n\n\n",
        "type": "AgentActionMessageLog",
        "message_log": [
          {
            "content": "",
            "additional_kwargs": {
              "tool_calls": [
                {
                  "id": "call_VoddpErSf67rwF8oGTP5TJoC",
                  "function": {
                    "arguments": "{\"code\":\"import pandas as pd\\n\\ndf = pd.read_csv('/home/user/data_analysis10063527483056799834.csv')\\nprint(df.head())\"}",
                    "name": "run_python_code"
                  },
                  "type": "function"
                }
              ]
            },
            "response_metadata": {
              "token_usage": {
                "completion_tokens": 46,
                "prompt_tokens": 378,
                "total_tokens": 424
              },
              "model_name": "gpt-4-turbo-preview",
              "system_fingerprint": "fp_1d2ae78ab7",
              "finish_reason": "tool_calls",
              "logprobs": null
            },
            "type": "ai",
            "id": "run-955cae96-5f50-42bf-9454-bea6591fedca-0",
            "example": false,
            "tool_calls": [
              {
                "name": "run_python_code",
                "args": {
                  "code": "import pandas as pd\n\ndf = pd.read_csv('/home/user/data_analysis10063527483056799834.csv')\nprint(df.head())"
                },
                "id": "call_VoddpErSf67rwF8oGTP5TJoC"
              }
            ],
            "invalid_tool_calls": []
          }
        ],
        "tool_call_id": "call_VoddpErSf67rwF8oGTP5TJoC"
      },
      [
        "  品牌名称        销售额\n0  周六福   70907697\n1  周大生   95379888\n2  周大福  167264148\n3  周生生   80441074",
        ""
      ]
    ]'''



'''"intermediate_steps": [
    [
      {
        "tool": "get_table_names",
        "tool_input": {},
        "log": "",
        "type": "AgentActionMessageLog",
        "message_log": [
          {
            "content": "",
            "additional_kwargs": {
              "tool_calls": [
                {
                  "id": "call_REoyFFuy8xhhiFRaKAhEvTYR",
                  "function": {
                    "arguments": "{}",
                    "name": "get_table_names"
                  },
                  "type": "function"
                }
              ]
            },
            "response_metadata": {
              "token_usage": {
                "completion_tokens": 11,
                "prompt_tokens": 365,
                "total_tokens": 376
              },
              "model_name": "gpt-4-turbo-preview",
              "system_fingerprint": "fp_79f643220b",
              "finish_reason": "tool_calls",
              "logprobs": null
            },
            "type": "ai",
            "id": "run-fde07b2a-2c7a-4048-ade6-d3802e79e938-0",
            "example": false
          }
        ]
      },
      "[('Album',), ('Artist',), ('Customer',), ('Employee',), ('Genre',), ('Invoice',), ('InvoiceLine',), ('MediaType',), ('Playlist',), ('PlaylistTrack',), ('Track',)]"
    ]
  ]'''
