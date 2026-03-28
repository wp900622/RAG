# RAG 專案

這是一個基於 LangChain 建立的 Retrieval-Augmented Generation (RAG) 測試與展示專案。本專案透過讀取 PDF 文件，將內容片段轉換為向量並存入 Milvus 向量資料庫，最後透過本地端執行的大型語言模型 (LLM) 提供基於文件的互動式問答 (QA) 功能。

## 專案結構與文件說明

本專案主要包含三個 Python 執行檔，分別展示了不同的功能模組：

### 1. `RAG.py` (核心問答系統)
這是專案的主程式，整合了完整的 RAG 工作流程，提供終端機互動介面。
- **文件處理**：使用 `PyPDFLoader` 讀取 PDF 檔案，並透過 `RecursiveCharacterTextSplitter` 進行文本切分。
- **向量資料庫與嵌入模型**：使用 HuggingFace 的 `sentence-transformers/all-MiniLM-L6-v2` 模型將文本轉為向量，並儲存至本地端運行的 **Milvus** 資料庫（預設埠為 `19530`）。
- **語言模型 (LLM)**：串接本地端執行的 **Ollama** 伺服器 (`http://localhost:11434`)，呼叫 `llama3` 模型。
- **自訂提示詞 (Prompt)**：設定了系統 Prompt，強制模型僅使用「繁體中文」回答使用者的問題。

### 2. `LLM.py` (Llama-cpp 測試腳本)
這是針對 LLM 單獨呼叫與 Prompt Template 的測試腳本。
- 使用 `llama_cpp` 載入位於本地端的量化 Llama 2 模型（`llama-2-7b-chat.Q2_K.gguf`）。
- 實作了 LangChain 的 `ConditionalPromptSelector`，當使用者提出問題時，讓模型產生 3 個相關聯的 Google 搜尋查詢推薦。

### 3. `LangChain.py` (Milvus 資料建置測試腳本)
獨立測試文件載入與寫入 Milvus 向量資料庫的流程。
- 使用 `PyMuPDFLoader` 載入 `Virtual_characters.pdf`。
- 文本切分與向量轉換測試。
- 展示了透過 `Milvus.from_documents` 方法將文件從記憶體存入資料庫內 (`langchain_example` collection) 的過程。

---

## 系統需求與環境設置

為確保專案順利運行，必須安裝並啟動以下外部系統服務：

### 1. 外部系統依賴
*   **Ollama**: 本機端 LLM 運行環境。請先下載並安裝 [Ollama](https://ollama.com/)，安裝後在終端機執行以準備 `llama3` 模型：
    ```bash
    ollama run llama3
    ```
*   **Milvus**: 向量資料庫。您需要在本地端背景運行 Milvus Server (通常可透過 [Docker Compose](https://milvus.io/docs/install_standalone-docker.md) 安裝，預設監聽 `localhost:19530`)。

### 2. Python 依賴套件
強烈建議建立 Python 虛擬環境後，安裝下列主要模組：
```bash
pip install langchain langchain-community langchain-core langchain-milvus
pip install pymilvus
pip install sentence-transformers
pip install pypdf pymupdf
pip install llama-cpp-python
```

---

## 執行與使用方式

確定 **Ollama** 模型伺服器以及 **Milvus** 向量資料庫 皆已啟動後：

1.  **檢查 PDF 檔案路徑**：
    若是直接執行 `RAG.py`，請先打開 `RAG.py`，確認 `main()` 函式裡的 `pdf_path` 變數指向正確的 PDF 檔案絕對位置。
    例如：將 `C:/Users/wp900/RAG/Virtual_characters.pdf` 改為您目錄下的 `./Virtual_characters.pdf` 等。
2.  **啟動問答腳本**：
    在終端機中執行：
    ```bash
    python RAG.py
    ```
3.  **開始對話**：
    腳本啟動後，會讀取並分析 PDF 內容，接著出現提示字元 `請輸入您的問題 (輸入 'quit' 結束):`。您可以輸入與文件內容相關的任何問題，系統將以繁體中文回答。輸入 `quit` 即可關閉對話。
