import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings()

data = pd.read_csv('data.csv')
question_to_source = Chroma(collection_name='quesiton_to_source', embedding_function=embedding_model, persist_directory='./rag_test')
query_to_thoughts = Chroma(collection_name= 'query_to_thoughts', embedding_function=embedding_model, persist_directory='./rag_test')



for i in range(len(data)):
    record = data.iloc[i,:]
    url_loc = record['url'].index('//')
    url = record['url'][url_loc:]
    # print(url)
    user_name = record['user_name']
    password = record['password']
    uri = f'mysql+mysqlconnector://{user_name}:{password}@{url[2:]}'
    database_name = uri[uri.rindex('/')+1:]
    question = record['name']
    thoughts = record['pre_req']
    sources = record['sources']
    id = record['id']
    sql = record['res_sql']
    question_to_source.add_texts(texts=[question],ids=[str(id)],metadatas=[{'source': sources, 'database':database_name}])
    query_to_thoughts.add_texts(texts=[thoughts], ids=[str(id)], metadatas=[{'sql':sql, 'database':database_name}])
    # break
    print(record['pre_req'])
    print(database_name)
    print(record['res_req'])
    print(record['name'])
    print('-'*30)
