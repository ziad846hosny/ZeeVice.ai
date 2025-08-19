# ZeeVice.AI

ZeeVice.AI is an experimental AI-powered **Graph RAG Search Engine** and **LLM Chat Assistant** project. It combines **graph-based retrieval**, **hybrid search**, and **LLM-powered reasoning** to deliver relevant responses over unstructured and structured data.

---

## 🚀 Project Overview

The project had two main components:

1. **Graph RAG Search Engine**

   * Built a **graph database** (Neo4j) to store entities, relationships, and metadata.
   * Scraped diverse unstructured data sources and ingested them into the graph.
   * Implemented **retrieval pipelines** to query the graph and augment LLM responses.

2. **LLM-Powered Assistant**

   * Integrated an LLM wrapper to process queries, generate natural responses, and use graph-based retrieval as context.
   * Designed a hybrid approach: **semantic search + graph query expansion**.

3. **Frontend Integration**

   * Developed a modern frontend with **React**, **TypeScript**, and **Vite** for speed and modularity.
   * Built a **chat-style UI** where users can interact with the assistant.

---

## 📊 Data Sources & Scraping

The following data was scraped, cleaned, and processed:

* **E-commerce products** (titles, specs, categories)
* **Technical specifications** (CPU, GPU, RAM, storage, OS, battery, etc.)
* **Brand & store information** for cross-source comparisons

Tools used for scraping:

* **Python + Selenium** (dynamic scraping)
* **Requests + BeautifulSoup** (static scraping)
* **Custom sanitization & parsing scripts**

---

## 🧠 Graph RAG Engine

* **Database**: Neo4j (GraphDB)
* **Pipeline**:

  * Entity & relationship extraction from scraped text
  * Graph-based retrieval with context expansion
  * Hybrid search combining keyword + semantic matching
* **Result**: More structured and relevant responses from the LLM

---

## 🤖 LLM Integration

* Used an **LLM wrapper** to process user queries
* Augmented responses with **graph-retrieved context**
* Implemented safeguards to reduce hallucinations (though not perfect 😉)

---

## 🛠️ Tech Stack

### **Frontend**

* **React** (UI components)
* **TypeScript** (type safety)
* **Vite** (fast build & dev environment)
* **shadcn/ui + TailwindCSS** (styling & components)

### **Backend**

* **Flask** (API layer)
* **Neo4j Driver** (graph queries)
* **OpenAI API / LLM API** (language model integration)

### **Data & Retrieval**

* **Selenium + Requests + BeautifulSoup** (scraping)
* **Neo4j Graph Database**
* **Hybrid Graph RAG pipeline**

---

## ⚡ How It Works

1. User enters a query in the chat UI
2. Backend routes the query:

   * → Graph Engine for retrieval
   * → LLM for reasoning & response generation
3. Augmented response is returned to the frontend

---

## 📌 Project Status

ZeeVice.AI has officially concluded.
This repo remains as a **reference implementation** of a Graph RAG + LLM system with a full-stack integration.

