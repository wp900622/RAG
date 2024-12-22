import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from pymilvus import connections, utility
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# 1. 設定 LLM 模型
def setup_llm():
    base_url="http://localhost:11434"
    llm = llm = Ollama(model="llama3", base_url="http://localhost:11434")
    return llm

# 2. PDF 處理與切分
def process_pdf(pdf_path):
    # 載入 PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # 切分文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

# 3. 設定向量資料庫
def setup_vector_store():
    # 連接 Milvus
    connect = connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    model_kwargs = {'device': 'cpu'}
    # 設定 embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = model_kwargs
    )
    
    return embeddings

# 4. 建立向量資料庫
def create_vector_store(chunks, embeddings):
    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={
            "host": "localhost",
            "port": "19530"
        },
        collection_name="pdf_store"
    )
    return vector_store

# 5. 建立 QA 鏈
def setup_qa_chain(vector_store, llm):

    retriever = vector_store.as_retriever()

    system_prompt = "現在開始使用我提供的情境來回答，只能使用繁體中文，不要有簡體中文字。如果你不確定答案，就說不知道。情境如下:\n\n{context}"
    prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "問題: {input}"),
    ]
)

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
# 主要功能整合
class PDFChatbot:
    def __init__(self, pdf_path):
        # 初始化 LLM
        self.llm = setup_llm()
        
        # 處理 PDF
        self.chunks = process_pdf(pdf_path)
        
        # 設定向量資料庫
        self.embeddings = setup_vector_store()
        
        # 建立向量資料庫
        self.vector_store = create_vector_store(
            self.chunks, 
            self.embeddings
        )
        
        # 設定 QA 鏈
        self.qa_chain = setup_qa_chain(
            self.vector_store, 
            self.llm
        )
    
    def ask_question(self, question):
        try:
            response = self.qa_chain.invoke(question)
            return response
        except Exception as e:
            return f"Error processing question: {str(e)}"

# 使用範例
def main():
    # 初始化聊天機器人
    pdf_path = "C:/Users/wp900/RAG/Virtual_characters.pdf"
    chatbot = PDFChatbot(pdf_path)
    context = []
    # 詢問問題
    while True:
        question = input("請輸入您的問題 (輸入 'quit' 結束): ")
        if question.lower() == 'quit':
            break
            
        response = chatbot.ask_question({"input": question, "context": context})
        context = response["context"]
        answer = response["answer"]
        print(f"回答:{answer} ")

if __name__ == "__main__":
    main()