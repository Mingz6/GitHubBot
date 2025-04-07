- [x] - Add a new space but fail everytime with the access token issue.
- [x] - Secrets management in new space
- [x] - Add RAG to the work --> https://huggingface.co/spaces/AFischer1985/AI-RAG-Interface-to-Hub/tree/main

- [ ] - ensure citation is available. 
- [ ] - add confidence score to the answer

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
PPT note:
For who
Use case
Demo
QRcode generator


- [ ] - Test with https://huggingface.co/datasets?task_categories=task_categories:question-answering&sort=trending

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

[Linkedin Post](https://www.linkedin.com/posts/pau-labarta-bajo-4432074b_machinelearning-mlops-realworldml-activity-7313470322568032256-Jo0x/):
5 years ago I struggled to land my first freelance ML engineering contract.
Then I discovered this ↓
  
  
Building one professional real-world ML project is the best way to stand out from the crowd, and land an ML job.

And here is how you can do it, 𝘀𝘁𝗲𝗽-𝗯𝘆-𝘀𝘁𝗲𝗽 👩💻👨🏽💻↓

𝗦𝘁𝗲𝗽 𝟭. 𝗙𝗶𝗻𝗱 𝗮 𝗿𝗲𝗮𝗹-𝘄𝗼𝗿𝗹𝗱 𝗽𝗿𝗼𝗯𝗹𝗲𝗺 𝘆𝗼𝘂 𝗮𝗿𝗲 𝗶𝗻𝘁𝗲𝗿𝗲𝘀𝘁𝗲𝗱 𝗶𝗻
Working on projects is harder than completing online courses.
But hey, no pain no gain.

You must work on a problem you are interested in.
Otherwise, you will quit.

𝗦𝘁𝗲𝗽 𝟮. 𝗙𝗶𝗻𝗱 𝗮 𝗱𝗮𝘁𝗮 𝘀𝗼𝘂𝗿𝗰𝗲
Preferably a live API. If not possible, pick a static dataset from Kaggle.

Here is a superb repo with a list of public APIs you can use
https://lnkd.in/dgUABbqt

𝗦𝘁𝗲𝗽 𝟯. 𝗕𝘂𝗶𝗹𝗱 𝗮 𝘀𝗶𝗺𝗽𝗹𝗲 𝗠𝗟 𝗺𝗼𝗱𝗲𝗹
Do not try to build THE PERFECT model, and only then move to the next phase.

Because this leads you to a never-ending Jupyter-notebook-development-cycle, and you get lost.

Start with basic features and a basic model.

And move to the next step.

𝗦𝘁𝗲𝗽 𝟰. 𝗕𝘂𝗶𝗹𝗱 𝗮 𝗠𝗶𝗻𝗶𝗺𝘂𝗺 𝗩𝗶𝗮𝗯𝗹𝗲 𝗣𝗿𝗼𝗱𝘂𝗰𝘁
A Jupyter notebook is not enough to prove your solution might work.
You need to go one step further and build a minimal working system.

I recommend you follow the 3-pipeline design ↓
https://lnkd.in/eWtTZEjT

𝗦𝘁𝗲𝗽 𝟱. 𝗦𝘁𝗮𝗿𝘁 𝗶𝘁𝗲𝗿𝗮𝘁𝗶𝗻𝗴 𝗼𝗻 𝘁𝗵𝗲 𝗺𝗼𝗱𝗲𝗹
Once the system works, start improving it by
- increasing training data size
- increasing the number of features
- trying a more complex ML model
- optimizing model hyper-parameters

𝗦𝘁𝗲𝗽 𝟲. 𝗣𝘂𝘀𝗵 𝘆𝗼𝘂𝗿 𝗰𝗼𝗱𝗲 𝘁𝗼 𝗮 𝗽𝘂𝗯𝗹𝗶𝗰 𝗚𝗶𝘁𝗛𝘂𝗯 𝗿𝗲𝗽𝗼 𝗮𝗻𝗱 𝘄𝗿𝗶𝘁𝗲 𝗮 𝗯𝗲𝗮𝘂𝘁𝗶𝗳𝘂𝗹 𝗥𝗘𝗔𝗗𝗠𝗘
The README file is the first thing your future employer will see.

Explain the problem you wanted to solve and the solution you built.

Here is an example
https://lnkd.in/eWkEUkwW