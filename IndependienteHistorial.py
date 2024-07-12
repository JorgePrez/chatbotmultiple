

# ------------------------------------------------------
# Streamlit
# Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗
# ------------------------------------------------------

import boto3
import logging

from typing import List, Dict
from pydantic import BaseModel
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory



import streamlit as st1


# ------------------------------------------------------
# Log level

##logging.getLogger().setLevel(logging.ERROR) # reduce log level

# ------------------------------------------------------
# Amazon Bedrock - settings

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}




# ------------------------------------------------------
# LangChain - RAG chain with chat history

prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, always answer in Spanish"
         "Answer the question based only on the following context:\n {context}"),
        MessagesPlaceholder(variable_name="history1"),
        ("human", "{question}"),
    ]
)

# Amazon Bedrock - KnowledgeBase Retriever 
retriever1 = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="ZTVA42XMAF", #  Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)

model1 = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

chain1 = (
    RunnableParallel({
        "context": itemgetter("question") | retriever1,
        "question": itemgetter("question"),
        "history1": itemgetter("history1"),
    })
    .assign(response = prompt1 | model1 | StrOutputParser())
    .pick(["response", "context"])
)

# Streamlit Chat Message History
history1 = StreamlitChatMessageHistory(key="chat_messages1")

# Chain with History
chain_with_history1 = RunnableWithMessageHistory(
    chain1,
    lambda session_id: history1,
    input_messages_key="question",
    history_messages_key="history1",
    output_messages_key="response",
)

# ------------------------------------------------------
# Pydantic data model and helper function for Citations

class Citation(BaseModel):
    page_content: str
    metadata: Dict

def extract_citations(response: List[Dict]) -> List[Citation]:
    return [Citation(page_content=doc.page_content, metadata=doc.metadata) for doc in response]

# ------------------------------------------------------
# S3 Presigned URL, esto permite realizar descargar del documento

def create_presigned_url(bucket_name: str, object_name: str, expiration: int = 300) -> str:
    """Generate a presigned URL to share an S3 object"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except NoCredentialsError:
        st1.error("AWS credentials not available")
        return ""
    return response

def parse_s3_uri(uri: str) -> tuple:
    """Parse S3 URI to extract bucket and key"""
    parts = uri.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key

# ------------------------------------------------------
# Streamlit


# Page title

st1.set_page_config(page_title='Chatbot CHH')

st1.subheader('Independiente (Hayek) 🔗', divider='rainbow')



# Clear Chat History function
def clear_chat_history():
    history1.clear()
    st1.session_state.messages4= [{"role": "assistant", "content": "Pregúntame sobre economía"}]

with st1.sidebar:
    st1.title('Independiente 🔗')
    streaming_on = st1.toggle('Streaming (Mostrar generación de texto en tiempo real)')
    st1.button('Limpiar pantalla', on_click=clear_chat_history)
    st1.divider()
    st1.write("History Logs")
    st1.write(history1.messages)

# Initialize session state for messages if not already present
if "messages4" not in st1.session_state:
    st1.session_state.messages4 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

# Display chat messages

for message in st1.session_state.messages4:
   with st1.chat_message(message["role"]):
       st1.write(message["content"])



# Chat Input - User Prompt 
if prompt := st1.chat_input():
    st1.session_state.messages4.append({"role": "user", "content": prompt})
    with st1.chat_message("user"):
        st1.write(prompt)

    config1 = {"configurable": {"session_id": "any"}}
    
    if streaming_on:
        # Chain - Stream
        with st1.chat_message("assistant"):
            placeholder1 = st1.empty()
            full_response1 = ''
            for chunk in chain_with_history1.stream(
                {"question" : prompt, "history1" : history1},
                config1
            ):
                if 'response' in chunk:
                    full_response1 += chunk['response']
                    placeholder1.markdown(full_response1)
                else:
                    full_context1 = chunk['context']
            placeholder1.markdown(full_response1)
            # Citations with S3 pre-signed URL
            citations1 = extract_citations(full_context1)
            with st1.expander("Mostrar fuentes >"):
                for citation in citations1:
                    st1.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)
                    presigned_url = create_presigned_url(bucket, key)
                    if presigned_url:
                        st1.markdown(f"Fuente: [{s3_uri}]({presigned_url})")
                    else:
                        st1.write(f"Fuente: {s3_uri} (Presigned URL generation failed)")
                    st1.write("**Score**:", citation.metadata['score'])
                    st1.write("--------------")

            # session_state append
            st1.session_state.messages4.append({"role": "assistant", "content": full_response1})
    else:
        # Chain - Invoke
        with st1.chat_message("assistant"):
            response = chain_with_history1.invoke(
                {"question" : prompt, "history1" : history1},
                config1
            )
            st1.write(response['response'])
            # Citations with S3 pre-signed URL
            citations = extract_citations(response['context'])
            with st1.expander("Mostrar fuentes >"):
                for citation in citations:
                    st1.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)
                    presigned_url = create_presigned_url(bucket, key)
                    if presigned_url:
                        st1.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                    else:
                        st1.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    st1.write("**Score**:", citation.metadata['score'])
                    st1.write("--------------")

            # session_state append
            st1.session_state.messages4.append({"role": "assistant", "content": response['response']})