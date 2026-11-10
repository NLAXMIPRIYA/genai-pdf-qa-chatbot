
# Development of a PDF-Based Intelligent Question-Answering System Using LangChain

## AIM

To develop a document-based question-answering system using LangChain that can read information from a PDF file, identify relevant content based on user queries, and generate accurate responses with the help of a Large Language Model.

## PROBLEM STATEMENT

Reading and searching through lengthy PDF documents manually can be time-consuming. The objective of this experiment is to build an intelligent chatbot that allows users to interact with a PDF document through natural-language questions.

The application retrieves relevant portions of the uploaded document and provides them as context to an LLM, which generates answers based on the retrieved information.

## TECHNOLOGIES AND TOOLS USED

- Python – Programming language
- LangChain – Framework for building LLM applications
- OpenAI Embeddings – Converts text into numerical representations
- GPT-4o-mini – Language model for generating answers
- ChromaDB – Vector database for storing document embeddings
- PyPDF – PDF text extraction
- Jupyter Notebook – Development and execution environment

## SYSTEM ARCHITECTURE

```text
       PDF Document
             |
             v
      PDF Text Extraction
             |
             v
       Text Segmentation
             |
             v
       Document Chunks
             |
             v
      Generate Embeddings
             |
             v
        ChromaDB
             |
             v
       Similarity Search
             |
             v
     Relevant Text Chunks
             |
             v
        GPT-4o-mini
             |
             v
       Generated Answer
```

## DESIGN AND IMPLEMENTATION

### Step 1: Extract Text from the PDF

The `PyPDFLoader` component of LangChain is used to read the selected PDF document. It extracts the textual content and organizes it into document objects containing page information and extracted text.

### Step 2: Divide the Extracted Text

The extracted content is separated into smaller sections using `RecursiveCharacterTextSplitter`.

The experiment uses a chunk size of 1000 characters with an overlap of 200 characters. This helps maintain the continuity of information between neighbouring chunks.

### Step 3: Generate Text Embeddings

The `OpenAIEmbeddings` model converts each text chunk into a numerical vector. These vectors represent the semantic meaning of the document content and help identify information related to a given question.

### Step 4: Store the Embeddings

The generated vectors and their corresponding document chunks are stored in ChromaDB. This vector database enables efficient retrieval of information based on semantic similarity.

### Step 5: Retrieve Relevant Information

A retriever is created using the vector database. Whenever a question is submitted, it searches the stored embeddings and returns the document chunks that are most relevant to the query.

### Step 6: Generate a Context-Based Response

The retrieved information is combined with the user's question and passed to GPT-4o-mini.

The language model processes the provided context and generates a meaningful response. The prompt instructs the model to use only the retrieved information while answering.

## PROGRAM

### 1. Import the Required Libraries

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

print("All required libraries imported successfully!")
```

### 2. Load the PDF Document

```python
loader = PyPDFLoader("GenoMatch_Project_Report (1).pdf")

documents = loader.load()

print("PDF loaded successfully!")
print("Total number of pages:", len(documents))
```

### 3. Examine the Extracted Content

```python
print("Length of extracted text:",
      len(documents[0].page_content))

print("\nExtracted content:")
print(documents[0].page_content[:1000])
```

### 4. Split the Document into Smaller Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Document splitting completed!")
print("Total number of chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)
```

### 5. Initialize the Embedding Model

```python
embeddings = OpenAIEmbeddings()

print("Embedding model initialized successfully!")
```

### 6. Create the Chroma Vector Database

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf_question_answering",
    persist_directory="pdf_chroma_db"
)

print("Chroma vector database created successfully!")
```

### 7. Configure the Document Retriever

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

print("Document retriever initialized!")
```

### 8. Retrieve Information Relevant to a Question

```python
question = "What is the primary goal of GenoMatch?"

relevant_docs = retriever.invoke(question)

print("Number of relevant chunks:",
      len(relevant_docs))

for i, doc in enumerate(relevant_docs):
    print(f"\n--- Retrieved Chunk {i+1} ---")
    print(doc.page_content)
```

### 9. Initialize the Language Model

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("Language model initialized successfully!")
```

### 10. Generate the Answer Using Retrieved Context

```python
context = "\n\n".join(
    doc.page_content for doc in relevant_docs
)

prompt = f"""
Answer the question using only the information
available in the given context.

If the context does not contain the answer,
state that the information is not available.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)

print("Generated Answer:")
print(response.content)
```

## OUTPUT

<img width="1080" height="397" alt="image" src="https://github.com/user-attachments/assets/e2c468f1-d0e0-4c5c-a827-013057860698" />


**Input Question:**

What is the primary goal of GenoMatch?

**Generated Response:**

The primary goal of GenoMatch is to classify noisy sequencing reads as Known, Unknown, or Insufficient Evidence, while providing ranked taxonomic candidates, confidence/evidence signals, and reference support.

*Note: Replace this sample response with the actual output obtained during execution.*

## RESULT

A PDF-based intelligent question-answering system was successfully developed using LangChain and OpenAI models. The application extracts information from PDF documents, divides the content into manageable chunks, generates semantic embeddings, and stores them in ChromaDB.

When a user submits a question, the retriever identifies relevant document sections and supplies them to GPT-4o-mini to generate a context-based response.

Thus, the experiment demonstrates how Retrieval-Augmented Generation (RAG) can be used to build an interactive question-answering application that makes information retrieval from PDF documents easier and more efficient.
