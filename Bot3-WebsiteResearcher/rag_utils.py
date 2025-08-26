from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os
import json

class RAGProcessor:
    def __init__(self, model_name=None):
        """Initialize the RAG processor with model options"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Don't initialize embeddings yet - do it lazily
        self.embeddings = None
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        
        # Storage for vectorstores
        self.vectorstores = {}
    
    def _initialize_embeddings(self):
        """Lazily initialize embeddings when needed with fallback options"""
        if self.embeddings is not None:
            return True
            
        try:
            # Try to initialize the primary embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )
            return True
        except Exception as e:
            print(f"Error loading primary embedding model: {str(e)}")
            try:
                # Fall back to a smaller, more commonly available model
                print("Attempting to load fallback embedding model...")
                self.embedding_model_name = "all-MiniLM-L6-v2"
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=f"sentence-transformers/{self.embedding_model_name}"
                )
                return True
            except Exception as e2:
                print(f"Error loading fallback embedding model: {str(e2)}")
                return False
        
    def process_website_content(self, website_url, content):
        """
        Process website content to create a searchable vectorstore
        
        Args:
            website_url: The URL of the website (used as an identifier)
            content: Dictionary with page URLs as keys and page content as values
        
        Returns:
            Success message
        """
        try:
            # Initialize embeddings if not already done
            if not self._initialize_embeddings():
                return {"error": "Failed to initialize embeddings model. Please check logs."}
                
            # Combine all page contents with source metadata
            documents = []
            
            for page_url, page_content in content.items():
                if isinstance(page_content, str) and page_content.strip():
                    # Split long content into chunks
                    chunks = self.text_splitter.split_text(page_content)
                    
                    # Add source metadata to each chunk
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "page_url": page_url, 
                            "content": chunk,
                            "chunk_id": i
                        })
            
            if not documents:
                return {"error": "No valid content to process"}
            
            # Extract just the text for embedding
            texts = [doc["content"] for doc in documents]
            metadatas = [{"source": doc["page_url"], "chunk_id": doc["chunk_id"]} for doc in documents]
            
            # Create vector store
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Store vectorstore using website URL as key
            self.vectorstores[website_url] = {
                "vectorstore": vectorstore,
                "document_count": len(documents)
            }
            
            return {
                "status": "success", 
                "message": f"Processed {len(documents)} document chunks from {len(content)} pages"
            }
            
        except Exception as e:
            return {"error": f"Error processing content: {str(e)}"}
    
    def retrieve_relevant_content(self, website_url, query, top_k=3):
        """
        Retrieve relevant content for a query from a processed website
        
        Args:
            website_url: URL of the previously processed website
            query: The user's question
            top_k: Number of relevant passages to retrieve
            
        Returns:
            List of relevant text passages and their sources
        """
        if website_url not in self.vectorstores:
            return {"error": "Website has not been processed yet"}
            
        try:
            vectorstore = self.vectorstores[website_url]["vectorstore"]
            retrieved_docs = vectorstore.similarity_search(query, k=top_k)
            
            results = []
            for doc in retrieved_docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown source")
                })
                
            return {
                "status": "success",
                "results": results
            }
                
        except Exception as e:
            return {"error": f"Error retrieving content: {str(e)}"}
    
    def clear_website_data(self, website_url=None):
        """
        Clear stored vectorstores for a specific website or all websites
        
        Args:
            website_url: URL to clear, or None to clear all
            
        Returns:
            Success message
        """
        if website_url:
            if website_url in self.vectorstores:
                del self.vectorstores[website_url]
                return {"status": "success", "message": f"Cleared data for {website_url}"}
            else:
                return {"error": "Website not found in storage"}
        else:
            self.vectorstores = {}
            return {"status": "success", "message": "Cleared all website data"}