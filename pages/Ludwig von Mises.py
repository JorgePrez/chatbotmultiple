

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

import streamlit as st3


# ------------------------------------------------------
# Log level

#logging.getLogger().setLevel(logging.ERROR) # reduce log level

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

prompt3 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, always answer in Spanish"
         "Answer the question based only on the following context:\n {context}"),
        MessagesPlaceholder(variable_name="history3"),
        ("human", "{question}"),
    ]
)

# Amazon Bedrock - KnowledgeBase Retriever 
retriever3 = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="4L0WE8NOOH", # Set your Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
)

model3 = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

chain3 = (
    RunnableParallel({
        "context": itemgetter("question") | retriever3,
        "question": itemgetter("question"),
        "history3": itemgetter("history3"),
    })
    .assign(response = prompt3 | model3 | StrOutputParser())
    .pick(["response", "context"])
)

# Streamlit Chat Message History
history3 = StreamlitChatMessageHistory(key="chat_messages3")

# Chain with History
chain_with_history3 = RunnableWithMessageHistory(
    chain3,
    lambda session_id: history3,
    input_messages_key="question",
    history_messages_key="history3",
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
        st3.error("AWS credentials not available")
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
st3.set_page_config(page_title='Chatbot CHH')
st3.subheader('Ludwig von Mises 🔗', divider='rainbow')

# Clear Chat History function
def clear_chat_history():
    history3.clear()
    st3.session_state.messages3 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

with st3.sidebar:
    st3.title('Ludwig von Mises 🔗')
    streaming_on = st3.toggle('Streaming (Mostrar generación de texto en tiempo real)',value=True)
    st3.button('Limpiar chat', on_click=clear_chat_history)
    st3.divider()
    st3.write("History Logs")
    st3.write(history3.messages)

# Initialize session state for messages if not already present
if "messages3" not in st3.session_state:
    st3.session_state.messages3 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

# Display chat messages
for message in st3.session_state.messages3:
    with st3.chat_message(message["role"]):
        st3.write(message["content"])

# Chat Input - User Prompt 
if prompt := st3.chat_input():
    st3.session_state.messages3.append({"role": "user", "content": prompt})
    with st3.chat_message("user"):
        st3.write(prompt)

    config3 = {"configurable": {"session_id": "any"}}
    
    if streaming_on:
        # Chain - Stream
        with st3.chat_message("assistant"):
            placeholder3 = st3.empty()
            full_response3 = ''
            for chunk in chain_with_history3.stream(
                {"question" : prompt, "history3" : history3},
                config3
            ):
                if 'response' in chunk:
                    full_response3 += chunk['response']
                    placeholder3.markdown(full_response3)
                else:
                    full_context3 = chunk['context']
            placeholder3.markdown(full_response3)
            # Citations with S3 pre-signed URL
            citations3 = extract_citations(full_context3)
            with st3.expander("Mostrar fuentes >"):
                for citation in citations3:
                    st3.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    
                    bucket, key = parse_s3_uri(s3_uri)
                    presigned_url = create_presigned_url(bucket, key)
                   ## if presigned_url:
                   ##         st3.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                   ## else:
                   ##         st3.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    st3.write(f"**Fuente**: *{key}* ")
                 
                    st3.write("**Score**:", citation.metadata['score'])
                    st3.write("--------------")

            # session_state append
            st3.session_state.messages3.append({"role": "assistant", "content": full_response3})
    else:
        # Chain - Invoke
        with st3.chat_message("assistant"):
            response = chain_with_history3.invoke(
                {"question" : prompt, "history3" : history3},
                config3
            )
            st3.write(response['response'])
            # Citations with S3 pre-signed URL
            citations = extract_citations(response['context'])
            with st3.expander("Mostrar fuentes >"):
                for citation in citations:
                    st3.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)
                  ##  presigned_url = create_presigned_url(bucket, key)
                  ##  if presigned_url:
                  ##          st3.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                  ##  else:
                  ##          st3.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    st3.write(f"**Fuente**: *{key}* ")
                
                    st3.write("**Score**:", citation.metadata['score'])
                    st3.write("--------------")

            # session_state append
            st3.session_state.messages3.append({"role": "assistant", "content": response['response']})