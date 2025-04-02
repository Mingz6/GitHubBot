- [x] - Add a new space but fail everytime with the access token issue.
- [x] - Secrets management in new space
- [x] - Add RAG to the work --> https://huggingface.co/spaces/AFischer1985/AI-RAG-Interface-to-Hub/tree/main

- [ ] - ensure citation is available. 

- [ ] - email Q&A based on database/website/documents -> https://huggingface.co/spaces/AimlAPI/EmailAgent
- [ ] - Chat features (Spical chat bot and store chat history)
- [ ] - PPT for demo
      - Motivation 
        - Pain Points -> Information Overloaded
        - Protential solution -> active learning
        - Why it matters -> Learn fast and more effortlessly
      - Why
        - No active search (Slow)
        - Inaccurate information
        - Hard to build custom DB (pro knowledge required)
        - Fast change and outdated DB cause issue
      - How
        - Input (txt, website, docs, pdf, etc)
        - Define agents and Roles
          - LLM Summary
          - LLM agent...
        - Define knowledge base (RAG)
        - Define output (email, chat, audio, etc)
      - Use cases (who)
        - Developers
        - Help Desks
      - Landing Page (Demo)

- [ ] - Answer faster and more accurate (Increase chrunck size or using multiple chrunks)

- [ ] - Code Review and Suggest --> https://github.com/semgrep/semgrep
- [ ] - deep research --> https://github.com/jina-ai/node-DeepResearch
- [ ] - Voice feature --> https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS

ToAsk:

Error answering question: Error code: 400 - {"message": "This model's maximum context length is 32768 tokens. However, you requested 104488 tokens (103464 in the messages, 1024 in the completion). Please reduce the length of the messages or completion.", "type_": "invalid_request_error"}

Make the rag using max_tokens 1000tokens limit --> Why? Less tokens means less cost and faster response time? but less knowledge?

- [ ] - Vector database
名称	适用场景	主要特点	部署方式
FAISS (Facebook AI Similarity Search)	本地向量搜索, 轻量级应用	超快查询速度, 适合小规模数据集	纯 Python 库，适用于本地部署
Milvus	AI 搜索, 语义搜索, 兼容大数据	支持分布式存储，提供高性能 ANN	可本地部署，也可云端使用
Weaviate	语义搜索, NLP 应用, 向量增强搜索	内置文本处理 (可直接处理 OpenAI/Transformers 生成的 embeddings)	支持 Docker 部署
Pinecone	SaaS 向量数据库, 免维护	高可用，适合云端应用，无需自行运维	只能云端使用，免费额度有限
Chroma	AI 辅助搜索, 个人知识管理	轻量级，适合本地 AI 代理应用	纯 Python 库，可本地或云端使用
