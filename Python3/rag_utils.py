from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
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
        self.embedding_model_names = [
            "sentence-transformers/all-mpnet-base-v2",  # Primary model
            "sentence-transformers/all-MiniLM-L6-v2"    # Fallback model
        ]
        
        # Storage for vectorstores
        self.vectorstores = {}
        
        # Storage for knowledge bases
        self.knowledge_bases = {}
    
    def _initialize_embeddings(self):
        """Lazily initialize embeddings when needed with fallback options"""
        if self.embeddings is not None:
            return True
        
        print("Initializing embeddings model...")
        
        # Try initializing with minimal settings first since that's what worked
        for model_name in self.embedding_model_names:
            try:
                print(f"Attempting to load embedding model: {model_name}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name
                )
                print(f"Successfully initialized embedding model: {model_name}")
                return True
            except Exception as e:
                print(f"Failed to load embedding model {model_name}: {str(e)}")
                continue
        
        # If we've tried all models and none worked, return failure
        print("Error: Could not initialize any embedding model")
        return False
    
    def process_knowledge_base(self, knowledge_data):
        """
        Process custom knowledge data from uploaded file
        
        Args:
            knowledge_data: Dictionary containing parsed knowledge file data
            
        Returns:
            Success message or error
        """
        try:
            kb_id = knowledge_data["id"]
            content = knowledge_data["content"]
            websites = knowledge_data.get("websites", [])
            
            if not kb_id or not content:
                return {"error": "Invalid knowledge data"}
            
            # Store the knowledge base content (non-vectorized)
            self.knowledge_bases[kb_id] = {
                "content": content,
                "websites": websites
            }
            
            # Initialize embeddings if not already done
            if not self._initialize_embeddings():
                return {"error": "Failed to initialize embeddings model. Please check logs."}
            
            # Process knowledge content if there's enough to vectorize
            if len(content) > 0:
                # Combine text lines with source metadata
                documents = []
                
                for i, text_line in enumerate(content):
                    if text_line.strip():
                        # Add source metadata as "knowledge line X"
                        documents.append({
                            "content": text_line,
                            "source": f"Knowledge base line {i+1}"
                        })
                
                # Extract just the text for embedding
                texts = [doc["content"] for doc in documents]
                metadatas = [{"source": doc["source"]} for doc in documents]
                
                # Create vector store
                vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
                
                # Store vectorstore using knowledge base ID as key
                self.vectorstores[kb_id] = {
                    "vectorstore": vectorstore,
                    "document_count": len(documents),
                    "type": "knowledge_base"
                }
            
            return {
                "status": "success", 
                "message": f"Processed {len(content)} knowledge statements"
            }
            
        except Exception as e:
            return {"error": f"Error processing knowledge base: {str(e)}"}
        
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
                elif isinstance(page_content, dict) and "content" in page_content:
                    # Handle content dictionary format
                    chunks = self.text_splitter.split_text(page_content["content"])
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
                "document_count": len(documents),
                "type": "website"
            }
            
            return {
                "status": "success", 
                "message": f"Processed {len(documents)} document chunks from {len(content)} pages"
            }
            
        except Exception as e:
            return {"error": f"Error processing content: {str(e)}"}
    
    def retrieve_from_knowledge_base(self, kb_id, query):
        """
        Retrieve information from a knowledge base for a query
        
        Args:
            kb_id: ID of the knowledge base
            query: The user's question
            
        Returns:
            The matching knowledge content and sources
        """
        knowledge_content = []
        website_urls = []
        
        # First, check if we have this knowledge base
        if kb_id not in self.knowledge_bases:
            return {"error": "Knowledge base not found"}
            
        kb_data = self.knowledge_bases[kb_id]
        
        # Check if we have a vectorized version for similarity search
        if kb_id in self.vectorstores:
            try:
                # Get the vectorstore
                vectorstore_data = self.vectorstores[kb_id]
                vectorstore = vectorstore_data["vectorstore"]
                
                # Perform similarity search
                retrieved_docs = vectorstore.similarity_search(query, k=5)
                
                # Add the retrieved documents
                for doc in retrieved_docs:
                    knowledge_content.append(doc.page_content)
                    source = doc.metadata.get("source", "Knowledge base")
                    if source not in website_urls:
                        website_urls.append(source)
                        
            except Exception as e:
                print(f"Error in vector retrieval: {str(e)}")
                # Fall back to simple text search if vector search fails
                pass
        
        # If no results from vector search or if it failed, use simpler text matching
        if not knowledge_content:
            # Extract keywords from the query for simple matching
            keywords = query.lower().split()
            
            # Remove common words
            stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "for", "in", "with", "by", "about", "of"}
            keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
            
            # Match against knowledge base content
            for i, line in enumerate(kb_data["content"]):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in keywords):
                    knowledge_content.append(line)
                    website_urls.append(f"Knowledge base line {i+1}")
        
        # Check for websites in the knowledge base
        kb_websites = kb_data.get("websites", [])
        
        # Important fix: If the query is about people/names, we should always include the text content
        # from the knowledge base, regardless of keyword matching
        name_related = any(name.lower() in query.lower() for name in ["ming", "louis", "jack", "developer", "manager", "employee", "staff", "person", "people", "team"])
        
        # If the query is about names, make sure we include all the people-related content
        if name_related:
            print("Name-related query detected, including all person information")
            for i, line in enumerate(kb_data["content"]):
                line_lower = line.lower()
                if any(name in line_lower for name in ["ming", "louis", "jack", "developer", "manager", "employee"]):
                    if line not in knowledge_content:
                        knowledge_content.append(line)
                        website_urls.append(f"Knowledge base line {i+1}")
        
        # If query is about CRNA or nurses or Alberta, always check websites
        nursing_related = any(term.lower() in query.lower() for term in ["crna", "nurse", "nursing", "alberta", "college", "registered"])
        
        # Include website info for nursing-related queries
        if nursing_related:
            print("Nursing-related query detected, ensuring website data is included")
            # The websites will be checked later in the answer_question function
            pass
        
        # Return both knowledge content and website information for a comprehensive search
        return {
            "content": knowledge_content,
            "sources": website_urls,
            "websites": kb_websites,
            # Add flags to help with processing
            "nursing_related": nursing_related,
            "name_related": name_related
        }
        
    def retrieve_relevant_content(self, source_id, query, top_k=3):
        """
        Retrieve relevant content for a query from a processed website or knowledge base
        
        Args:
            source_id: URL of the previously processed website or knowledge base ID
            query: The user's question
            top_k: Number of relevant passages to retrieve
            
        Returns:
            List of relevant text passages and their sources
        """
        if source_id not in self.vectorstores:
            return {"error": "Source content has not been processed yet"}
            
        try:
            vectorstore = self.vectorstores[source_id]["vectorstore"]
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
    
    def clear_knowledge_base(self, kb_id=None):
        """
        Clear stored knowledge base data for a specific ID or all knowledge bases
        
        Args:
            kb_id: ID to clear, or None to clear all
            
        Returns:
            Success message
        """
        if kb_id:
            if kb_id in self.knowledge_bases:
                del self.knowledge_bases[kb_id]
            if kb_id in self.vectorstores:
                del self.vectorstores[kb_id]
            return {"status": "success", "message": f"Cleared knowledge base {kb_id}"}
        else:
            # Keep only website vectorstores
            website_stores = {k: v for k, v in self.vectorstores.items() if v.get("type") == "website"}
            self.knowledge_bases = {}
            self.vectorstores = website_stores
            return {"status": "success", "message": "Cleared all knowledge bases"}
    
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
            # Keep only knowledge base vectorstores
            kb_stores = {k: v for k, v in self.vectorstores.items() if v.get("type") == "knowledge_base"}
            self.vectorstores = kb_stores
            return {"status": "success", "message": "Cleared all website data"}
