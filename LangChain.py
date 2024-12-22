from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("Virtual_characters.pdf")
PDF_data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)


from langchain.embeddings import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs)

from langchain_milvus import Milvus

# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
URI = "http://localhost:19530"

vector_store = Milvus(
    embedding_function=embedding,
    connection_args={"uri": URI},
)


from langchain_core.documents import Document

vector_store_saved = Milvus.from_documents(
    [Document(page_content="foo!")],
    embedding,
    collection_name="langchain_example",
    connection_args={"uri": URI},
)

