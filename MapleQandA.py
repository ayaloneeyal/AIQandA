from dotenv import load_dotenv

load_dotenv()
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import StructuredTool

import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime


im = Image.open("./favicon.ico") 
st.set_page_config(page_title="Maple AI assistent", page_icon=im)
st.title(":maple_leaf: Maple AI assistent")
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

@st.cache_resource
def configure_retriever():

    text = ""
    with open("./OriginalMapleGeneralQandA.txt") as f:
        text = f.read()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4}
    )

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        self.container.write(documents)



def get_stock_price(stock_name: str, start_date : str, end_date: str) -> str:
    """Searches the stock price for a given time period
       query must provide stock_name start_date and end_date.
       
       stock_name is the stock name
       start date is the starting date in the format of yyyy-mm-dd
       end date is the  ending date in the format of yyyy-mm-dd
       
       The answer from this tool is final and must be used.
    """
   
    start_date =  datetime.datetime.strptime(start_date,"%Y-%m-%d")
    end_date =  datetime.datetime.strptime(end_date,"%Y-%m-%d")
    
    tickerData = yf.Ticker(stock_name) # Get ticker data
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

    delta = len(tickerDf)
    delta = int(delta/2)
    if delta > 20 :
        delta = 20

    # Bollinger bands
    st.header('**Bollinger Bands**')
    qf=cf.QuantFig(tickerDf,kind='candlestick', title=f"{stock_name} price",legend='top',name=f"{stock_name}")
    qf.add_bollinger_bands(periods=delta,boll_std=2,colors=['magenta','grey'],fill=True)
    qf.add_volume(name='Volume',up_color='green', down_color='red')
    fig = qf.iplot(asFigure=True)
   
    st.plotly_chart(fig)
    close_price = tickerDf['Close'].iat[-1]

    return f"{stock_name} closing price was on the last date was {close_price}"

@st.cache_data
def search_knowledgebase(query: str) -> str:
    """Use this tool when answering questions about Maple internal knowledgebase..
       
    """
    retriever = configure_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa.run(query)    


tool1 = StructuredTool.from_function(get_stock_price)
tool2 = StructuredTool.from_function(search_knowledgebase)

@st.cache_resource
def llm_chain_response():

    tools = [tool1,
             tool2]
             
             
             
            # Tool(
            #     name="Knowledge Base",
            #     func=qa.run,
            #     description=(
            #         "use this tool when answering questions about Maple internal knowledgebase."
            #         "Always use this tool first to see if you can get an answer"
            #     ),
            # ),
            # Tool(
            #     name="Get stock price",
            #     func=StructuredTool.from_function(get_stock_price),
            #     description=(
            #         """Use this tool to show the stock prices between 2 dates.
            #         you must provide 3 string, the stock_name is the name, start_date is the starting date, and end_date is the ending date
            #         the answer from that tool should always be considered as the final answer to the question"""
            #     ),
            # )



    PREFIX = """Answer the following questions as accurate as you can, but speaking in rhymes. 
                Always use the knowledge base tool when answering questions. 
                you have no knowledge about stock prices.
                current year is 2023.
                You have access to the following tools:"""
                
                
    system_message = SystemMessage(
          content="""Answer the following questions as accurate as you can, but speaking a passionate salesperson. 
                Always use the knowledge base tool when answering questions related to maple. Dont guess!. 
                you have no knowledge about stock prices.
                current year is 2023."""
      )                
    #memory = ConversationBufferMemory(memory_key="chat_history", k=5, return_messages=True)
    
    # agent_kwargs = {
    # "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    # }
    chat_history = [MessagesPlaceholder(variable_name="memory")]

    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    
    # chat_history = MessagesPlaceholder(variable_name="chat_history")
    # memory = ConversationBufferMemory(memory_key="chat_history", k=10,return_messages=True)
    
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0613",
        temperature=0,
        streaming=True,
    )


    agent = initialize_agent(
        agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors="Check your output and make sure it conforms!",
        early_stopping_method="generate",
        memory=memory,
        agent_kwargs={
            "system_message": system_message,
            "extra_prompt_messages": chat_history,
            },
    )
 
    return agent



if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "assistant" :
        st.chat_message(msg["role"],avatar='https://raw.githubusercontent.com/ayaloneeyal/AIQandA/main/favicon.ico').write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    agent =  llm_chain_response()

    with st.chat_message("assistant",avatar='https://raw.githubusercontent.com/ayaloneeyal/AIQandA/main/favicon.ico'):
        # retrieval_handler = PrintRetrievalHandler(st.container())
        # stream_handler = StreamHandler(st.empty()   )
        # , callbacks=[retrieval_handler,stream_handler]
        response = agent.run(user_query)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.container().write(response)
