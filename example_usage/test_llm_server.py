import os

api_base = "https://llm-gateway.k8s-test.cs.aalto.fi/v1"
api_key = "321"
model_name = "llama2-7b"
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = api_base

from langchain.docstore.document import Document
from langchain.embeddings.infinity import InfinityEmbeddings
from langchain.text_splitter import CharacterTextSplitter


documents = [Document(page_content="Hello world!", metadata={"source": "local"})]

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
from SimpleEmbeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model=model_name)

embeds = embeddings.embed_documents([doc.page_content for doc in docs])
print(embeds)

# from langchain_community.embeddings import openai

# api_base = "https://llm-gateway.k8s-test.cs.aalto.fi/v1"
# api_key = "321"
# model_name = "llama2-7b"
# embeddings = openai.OpenAIEmbeddings(
#    model=model_name, openai_api_base=api_base, openai_api_key=api_key
# )
# text = "Hello lets test this"
# doc_result = embeddings.embed_documents([text])
