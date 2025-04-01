- [x] - Add a new space but fail everytime with the access token issue.
- [x] - Secrets management in new space
- [x] - Add RAG to the work --> https://huggingface.co/spaces/AFischer1985/AI-RAG-Interface-to-Hub/tree/main

- [ ] - ensure citation is available. 

- [ ] - email Q&A based on database/website/documents -> https://huggingface.co/spaces/AimlAPI/EmailAgent
- [ ] - Chat features
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

- [ ] - Code Review and Suggest --> https://github.com/semgrep/semgrep
- [ ] - deep research --> https://github.com/jina-ai/node-DeepResearch
- [ ] - Voice feature --> https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS

ToAsk:
Make the rag using max_tokens 1000tokens limit --> Why? Less tokens means less cost and faster response time? but less knowledge?
What makes difference Deep research vs duckduckgo search vs crawling?
How to store the knowledge base (database, **json**, or something else)
vector database? Chrome DB for RAG embedding.
- [ ] - Consider using a vector database for RAG embedding.
Answer faster and more accurate (Increase chrunck size or using multiple chrunks)

Email feature -> Learn from previous emails examples
Chat feature -> Create a chat bot using LLM (Spical chat bot and store chat history)
Voice feature -> Speech to text and text to speech
