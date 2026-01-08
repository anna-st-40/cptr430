---
marp: true
theme: default
paginate: true
---

# CPTR 430: Artificial Intelligence
## Winter 2026

**Instructor:** Chiké Abuah  
**Schedule:** M/W/Th/F, 10:00–10:50 AM  
**Location:** KRH 327

---

# What is AI?

> "The science and engineering of making intelligent agents"
> — Russell & Norvig

- **Agents** perceive their environment and take actions
- AI systems reason, plan, learn, and solve problems
- Applications span from search engines to self-driving cars

**This course:** Build practical AI systems using modern techniques

---

# Course Structure

- **50% Programming Labs** – Apply concepts hands-on
- **50% Course Project** – Explore AI in a specific domain
- **No final exam** – Project presentation instead
- **Project proposal due:** End of Week 2 (Jan 16)

**Focus:** Building AI applications, not training models from scratch

---

# Week 1-2: Search & Problem Solving

**Topics:** Uninformed search, informed search, heuristics, game playing

**Why it matters:**
- Foundation for pathfinding (GPS, robotics, games)
- Introduces state spaces and optimization
- Core technique: A* search still powers many systems

**Lab:** Implement search algorithms for real problems

---

# Week 3: Constraint Satisfaction

**Topics:** CSP algorithms, backtracking, arc consistency

**Why it matters:**
- Scheduling problems (courses, flights, resources)
- Configuration problems (Sudoku, map coloring)
- Efficient techniques for combinatorial problems

**Application:** Many real-world problems are CSPs in disguise

---

# Week 4: Logic & Knowledge Representation

**Topics:** Propositional logic, first-order logic, inference

**Why it matters:**
- Represent structured knowledge explicitly
- Enable logical reasoning and proof
- Foundation for expert systems and rule-based AI

**Connection:** Bridges to modern knowledge graphs

---

# Weeks 5-6: Probability & Bayesian Networks

**Topics:** Probability theory, Bayesian networks, inference, HMMs

**Why it matters:**
- Handle uncertainty in the real world
- Medical diagnosis, spam filtering, speech recognition
- Foundation for machine learning

**Modern relevance:** Probabilistic reasoning underpins many ML systems

---

# Week 7: Automated Planning

**Topics:** STRIPS, planning algorithms, plan representation

**Why it matters:**
- Robotics and autonomous systems
- Task automation and workflow optimization
- AI agents that achieve complex goals

**Example:** Planning trips, manufacturing schedules, game AI

---

# Week 8-9: Decision Making & Learning

**Topics:** MDPs, sequential decision making, reinforcement learning

**Why it matters:**
- Learn optimal behavior from experience
- Powers game-playing AI (AlphaGo) and robotics
- Foundation for modern AI alignment research

**Bridge to today:** RL in chatbots and recommendation systems

---

# Week 9: NLP & Computer Vision

**Topics:** Natural language processing, computer vision (selective)

**Why it matters:**
- Understanding and generating text
- Image recognition and analysis
- Core components of modern AI applications

**Context:** How classical AI connects to deep learning era

---

# Week 10: Integration & Presentations

- Finalize projects
- Student presentations (Wed–Thu)
- Course reflection and wrap-up

**Goal:** Showcase your AI system and what you learned

---

# Course Project!

**Core principle:** AI as system integration

- Use existing tools (LLMs, retrieval, APIs)
- Combine reasoning, planning, and knowledge
- Build something that works

**Don't:** Train models from scratch  

---

# Project Timeline

- **Week 2 (Jan 16):** Project proposal due
- **Week 4 (Jan 30):** Architecture & data design
- **Weeks 5-6:** Midpoint demo
- **Week 8:** Final demo & written report
- **Week 10:** Presentations

**Fridays = Project Days:** Group work time with instructor support

---

# Project Idea 1: Course-Specific AI Tutor

**What:** Chatbot that answers questions using course materials

**AI concepts:** RAG, knowledge representation, prompt engineering

**Tech stack:** LLM + vector database + retrieval

**Why interesting:**
- Practical immediate application
- Teaches retrieval-augmented generation
- Requires handling uncertainty and citations

---

# Project Idea 2: AI Study Planner

**What:** Agent that creates personalized study schedules

**AI concepts:** Planning, constraint satisfaction, goal-based agents

**Tech stack:** LLM with tool calling + calendar API

**Why interesting:**
- Combines planning with user modeling
- Requires reasoning about constraints
- Demonstrates autonomous agents

---

# Project Idea 3: Technical Documentation Q&A

**What:** System for querying complex manuals (APIs, Linux docs)

**AI concepts:** Semantic search, knowledge extraction, explanation

**Tech stack:** Embeddings + hierarchical retrieval + LLM

**Why interesting:**
- Emphasis on accuracy and citations
- Compares different retrieval strategies
- Teaches structured information handling

---

# Project Idea 4: Codebase Navigator

**What:** Tool to explain and search large codebases

**AI concepts:** Symbolic + neural reasoning, structured search

**Tech stack:** Code parser + embeddings + LLM explanations

**Why interesting:**
- Combines code understanding with natural language
- Requires preventing hallucinated APIs
- Real developer pain point

---

# Project Idea 5: Multi-Agent Debate System

**What:** Multiple AI agents debate decisions from different viewpoints

**AI concepts:** Multi-agent systems, game theory, coordination

**Tech stack:** Multiple LLM instances with role prompts + judge agent

**Why interesting:**
- Explores agent interaction and conflict
- Demonstrates emergent behavior
- Connects to alignment research

---

# Project Idea 6: Autonomous Task Agent

**What:** Agent that completes bounded tasks (trip planning, event organizing)

**AI concepts:** Rational agents, planning, tool use, feedback loops

**Tech stack:** LLM with function calling + external tools

**Why interesting:**
- End-to-end autonomous system
- Requires error recovery
- Demonstrates agentic AI in practice

---

# More Project Ideas Available

- Domain-specific research assistant
- AI policy/compliance assistant
- Intelligent campus resource finder
- AI-powered debugging assistant
- And more...

**See full project ideas document** for details, stretch goals, and implementation guidance

---

# What Makes a Good Proposal?

**Include:**
1. Clear problem statement
2. AI techniques you'll apply
3. Expected inputs/outputs
4. Success criteria
5. Potential challenges
6. Timeline with milestones

**Due:** Friday, January 16 (end of Week 2)

---

# Support & Resources

**Office Hours:** KRH 329  
**Project Days:** Every Friday for hands-on help 
**AI Tools:** Lab Ollama Server, ai.cs tool

---

# Why This Course Matters

Classical AI techniques remain relevant:
- Search powers GPS and game AI
- Planning enables robotics
- Bayesian reasoning handles uncertainty
- Logic systems ensure safety-critical decisions

**Modern AI builds on these foundations**

---

# Keys to Success

1. **Attend class** – Core concepts build on each other
2. **Start projects early** – Integration takes time
3. **Use Fridays** – Project days are for your benefit
4. **Ask questions** – AI is complex; collaboration helps
5. **Think systems** – You're building intelligent agents

---

# Course Philosophy

**AI as Engineering:**
- Combine multiple techniques
- Handle real-world messiness
- Make design tradeoffs
- Consider ethical implications

**You're not just learning about AI—you're building AI systems**

---

# Looking Ahead

- **This week:** Foundations and search
- **Next week:** Heuristics and constraint satisfaction
- **Week 2 end:** Project proposals due
- **Weeks 3-9:** Deep dive into AI techniques
- **Week 10:** Show off what you built

**Let's build intelligent agents together!**

---

# Questions?

**Contact:**
- Email: chike.abuah@wallawalla.edu
- Office: KRH 329

**Resources:**
- Brightspace: [class.wallawalla.edu](class.wallawalla.edu)
- Textbook: Russell & Norvig, AI 4th edition