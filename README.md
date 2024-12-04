# **Drug Information Chatbot**

A conversational chatbot designed to provide information about drugs, their uses, alternatives, and recommendations. This project uses **LangGraph**, **ChromaDB**, and **ChatGroq** to build a dynamic workflow that processes user queries effectively.

---

## **Features**

- **Drug Information Retrieval**: Fetch details about drugs from a pre-built database.
- **Recommendations**: Provide advice based on drug usage queries.
- **Alternatives**: Suggest substitutes for a specific drug.
- **Summarization**: Summarize detailed information about a drug.

---

## **Installation**

### **1. Clone the Repository**
```bash git clone https://github.com/MelvinAGomez/Drug_Information_Chatbot.git```
cd Drug_Information_Chatbot
### **2. Set Up a Virtual Environment**
```bash python -m venv env
source env/bin/activate   # On Linux/Mac
env\Scripts\activate      # On Windows
```

### **3. Install Required Dependencies**
```bash
langchain==0.0.205       # For LangGraph and workflow management

langchain-core==0.0.9    # For managing core LangChain functionalities

langchain-groq          # For ChatGroq LLM integration

langchain-community     # For ChromaDB vectorstore integration

chromadb==0.3.23        # For ChromaDB as a vector database

streamlit==1.24.1       # For building the interactive chatbot UI

```

### **4. Set Up Environment Variables**
Create a file named secrets.env in the ./secrets folder and add your Groq API key:
```bash GROQ_API_KEY=your_groq_api_key```

## **How to Run**
### **1. Start the ChromaDB Server**
Build the database by running the following command:

```bash python chroma.py```

This script:
Reads data (e.g., drug information) from a dataset.
Stores the data into ChromaDB for retrieval during chatbot interactions.

## **2. Start the Chatbot**
Launch the chatbot application using Streamlit:

```bash streamlit run app.py ```

This opens a web interface where you can interact with the chatbot.

## **How It Works**
LangGraph Workflow
LangGraph is used to define a workflow of nodes (e.g., supervisor, question_answering, recommendation, etc.).
A supervisor node decides which worker node to route the query to based on its type:

Factual Queries → question_answering

Advice Requests → recommendation

Substitute Suggestions → alternatives

Summarization Requests → summarization


Each worker node processes the query using ChromaDB and an LLM (ChatGroq) for enhanced responses.
Retrieval with ChromaDB
ChromaDB is a vector database used to store and retrieve drug information.


### **During a query:**
The query is converted into an embedding.
ChromaDB performs a similarity search to find the most relevant data.
The retrieved data is sent to the LLM for final processing and formatting.


## **Key Dependencies**
LangGraph: For managing the workflow of query processing.

ChromaDB: For storing and retrieving drug information.

ChatGroq: For advanced natural language processing.

Streamlit: For building the interactive chatbot interface.


## **Sample Queries**

What is Aspirin?

Can I take Ibuprofen for a headache?

Suggest alternatives to Paracetamol.

Summarize the uses of Amoxicillin.
