#from dotenv import load_dotenv

#load_dotenv()
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
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


im = Image.open("./favicon.ico") 
st.set_page_config(page_title="Maple AI assistent", page_icon=im)
st.title(":maple_leaf: Maple AI assistent")


@st.cache_resource(ttl="1h")
def configure_retriever():

    text = ""
    with open("./OriginalMapleGeneralQandA.txt") as f:
        text = f.read()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
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


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


retriever = configure_retriever()
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
)

tools = [
        Tool(
            name="Knowledge Base",
            func=qa.run,
            description=(
                "use this tool when answering questions about our internal knowledge base"
                "Always use this tool first to see if you can get an answer"
            ),
        )
    ]

PREFIX = """Answer the following questions as best you can, but speaking as passionate Certified Financial Planner. Always use the Knowledge Base tool before doing a general search. You have access to the following tools:"""
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0,
    streaming=True,
)


agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    openai_api_key=openai_api_key,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors="Check your output and make sure it conforms!",
    early_stopping_method="generate",
    memory=memory,
    agent_kwargs={"prefix": PREFIX},
)



if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "assistant" :
        st.chat_message(msg["role"],avatar='https://raw.githubusercontent.com/dataprofessor/streamlit-chat-avatar/master/bot-icon.png').write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    from langchain.callbacks import StreamlitCallbackHandler


    with st.chat_message("assistant",avatar=im):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = agent.run(
            user_query)
        #, callbacks=[StreamlitCallbackHandler(st.container()),retrieval_handler]    )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant",avatar='https://raw.githubusercontent.com/dataprofessor/streamlit-chat-avatar/master/bot-icon.png').write(response])
