

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
import streamlit as st2


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

prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, always answer in Spanish"
         "Answer the question based only on the following context:\n {context}"),
        MessagesPlaceholder(variable_name="history2"),
        ("human", "{question}"),
    ]
)

# Amazon Bedrock - KnowledgeBase Retriever 
retriever2 = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="7MFCUWJSJJ", # Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
)

model2 = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

chain2 = (
    RunnableParallel({
        "context": itemgetter("question") | retriever2,
        "question": itemgetter("question"),
        "history2": itemgetter("history2"),
    })
    .assign(response = prompt2 | model2 | StrOutputParser())
    .pick(["response", "context"])
)

# Streamlit Chat Message History
history2 = StreamlitChatMessageHistory(key="chat_messages2")

# Chain with History
chain_with_history2 = RunnableWithMessageHistory(
    chain2,
    lambda session_id: history2,
    input_messages_key="question",
    history_messages_key="history2",
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
        st2.error("AWS credentials not available")
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
st2.set_page_config(page_title='Chatbot CHH')

st2.subheader('Henry Hazlitt 🔗', divider='rainbow')



# Clear Chat History function
def clear_chat_history():
    history2.clear()
    st2.session_state.messages2 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

with st2.sidebar:
    st2.title('Henry Hazlitt 🔗')
    streaming_on = st2.toggle('Streaming (Mostrar generación de texto en tiempo real)',value=True)
    st2.button('Limpiar chat', on_click=clear_chat_history)
    st2.divider()
    st2.write("History Logs")
    st2.write(history2.messages)

# Initialize session state for messages if not already present
if "messages2" not in st2.session_state:
    st2.session_state.messages2 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

# Display chat messages
for message in st2.session_state.messages2:
    with st2.chat_message(message["role"]):
        st2.write(message["content"])

# Chat Input - User Prompt 
if prompt := st2.chat_input():
    st2.session_state.messages2.append({"role": "user", "content": prompt})
    with st2.chat_message("user"):
        st2.write(prompt)

    config2 = {"configurable": {"session_id": "any"}}
    
    if streaming_on:
        # Chain - Stream
        with st2.chat_message("assistant"):
            placeholder2 = st2.empty()
            full_response2 = ''
            for chunk in chain_with_history2.stream(
                {"question" : prompt, "history2" : history2},
                config2
            ):
                if 'response' in chunk:
                    full_response2 += chunk['response']
                    placeholder2.markdown(full_response2)
                else:
                    full_context2 = chunk['context']
            placeholder2.markdown(full_response2)
            # Citations with S3 pre-signed URL
            citations2 = extract_citations(full_context2)
            with st2.expander("Mostrar fuentes >"):
                for citation in citations2:
                    st2.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    st2.write(f"**Fuente**: *{key}* ")

                    st2.write("**Score**:", citation.metadata['score'])
                    st2.write("--------------")

            # session_state append
            st2.session_state.messages2.append({"role": "assistant", "content": full_response2})
    else:
        # Chain - Invoke
        with st2.chat_message("assistant"):
            response = chain_with_history2.invoke(
                {"question" : prompt, "history2" : history2},
                config2
            )
            st2.write(response['response'])
            # Citations with S3 pre-signed URL
            citations = extract_citations(response['context'])
            with st2.expander("Mostrar fuentes >"):
                for citation in citations:
                    st2.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)
        
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    st2.write(f"**Fuente**: *{key}* ")
                    
                    st2.write("**Score**:", citation.metadata['score'])
                    st2.write("--------------")

            # session_state append
            st2.session_state.messages2.append({"role": "assistant", "content": response['response']})