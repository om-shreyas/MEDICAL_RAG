# Medical RAG Query System  

This project provides a Retrieval-Augmented Generation (RAG) model designed to query medical journals for solving medical-related queries. By combining document retrieval with advanced language models, the system generates accurate, contextually relevant answers referencing the supplied medical literature.  

---

## Features  

1. **Medical Journal Ingestion**  
   - Automatically extracts content from PDF-format medical journals stored in the `data` directory.  
   - Splits text into manageable chunks for optimized retrieval and indexing.  

2. **RAG Framework**  
   - Integrates document retrieval with generative AI to answer medical queries.  
   - Ensures factual, context-aware responses by grounding answers in referenced journals.  

3. **Chroma Vector Database**  
   - Efficiently stores document embeddings for similarity searches.  
   - Supports incremental updates to include new journals or reset the database entirely.  

4. **Web and API Interface**  
   - Flask-based API for querying the system programmatically.  
   - Web interface for users to interact with the system intuitively.

---

## Use Cases  

- **Medical Practitioners**: Quickly reference key insights from journals to support decision-making.  
- **Researchers**: Retrieve specific information related to a query for their studies.  
- **Students**: Access concise answers to medical questions, backed by journal citations.  

---

## Installation  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/medical-rag-system.git
   cd medical-rag-system
   ```  

2. **Set Up a Virtual Environment**:  
   ```bash
   python3 -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  
   ```  

3. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt  
   ```  

---

## File Structure  

```
medical-rag-system/
├── data/                     # Directory for input medical journal PDFs.
├── chroma/                   # Directory for the Chroma database.
├── templates/
│   └── chat.html             # Web interface for queries.
├── POPULATE_DATA.py          # Script to load and index medical journals.
├── GET_EMBEDDING_FUNCTION.py # Utility for embedding functions tailored for medical journals.
├── QUERY_DATA.py             # Script for querying the indexed database.
├── QUERY_API.py              # Flask-based API for query handling.
├── requirements.txt          # Python dependencies.
└── README.md                 # Documentation.
```  

---

## Usage  

### 1. Populate the Database  

Place your medical journals (PDF format) in the `data` directory.  

#### Populate or Update the Database:  
```bash
python POPULATE_DATA.py  
```  

#### Reset the Database:  
To clear and rebuild the database:  
```bash
python POPULATE_DATA.py --reset  
```  

---

### 2. Query via CLI  

Run the following command to query the database with a medical question:  
```bash
python QUERY_DATA.py "What are the latest treatments for type 2 diabetes?"  
```  

---

### 3. Launch the Web Interface  

Start the Flask application for the web interface:  
```bash
python QUERY_API.py  
```  

Visit `http://127.0.0.1:5000` in your browser to interact with the system.  

---

## How It Works  

1. **Document Processing**  
   - Journals in PDF format are parsed into text chunks.  
   - Each chunk is assigned metadata, including source, page number, and chunk ID, for easy identification.  

2. **Embedding and Storage**  
   - Text chunks are embedded using the `OllamaEmbeddings` model.  
   - The embeddings are stored in a persistent Chroma vector database, enabling similarity-based retrieval.  

3. **Query Handling**  
   - Queries are matched against the database to retrieve the most relevant text chunks.  
   - A response is generated using the retrieved context and the `mistral` language model.  

4. **Web Interface**  
   - The web-based chat interface simplifies user interaction for medical queries.  

---

## Key Dependencies  

- **LangChain**: For efficient document handling, text splitting, and embedding integration.  
- **Ollama**: Embedding generator and RAG response handler.  
- **Chroma**: Vector database for fast similarity-based lookups.  
- **Flask**: For API hosting and web interface.  

---

## Extending the System  

1. **Additional Data Formats**: Support formats beyond PDFs, like `.docx` or `.txt`, by updating the document loader in `GET_EMBEDDING_FUNCTION.py`.  
2. **Improved Language Models**: Integrate domain-specific models for better medical query responses.  
3. **Enhanced UI**: Upgrade `templates/chat.html` for a more user-friendly experience.  

---

## Contributing  

Contributions are welcome! Submit feature requests, report issues, or create pull requests to help improve this project.  

---

## License  

This project is licensed under the MIT License. See the LICENSE file for details.