# Project Proposal: Computer Science AI Advisor

## Problem Statement
Currently, every academic department has advisors that students talk to when they need
guidance as to what classes to take, when to take them, what requirements they are missing, etc.
The advisors rely on guidebooks, degree audits, 4-year plans. On the other side,
students frequently struggle to visualize how their remaining classes fit into a timeline, and they often miss prerequisites for some classes they need
to take or they fail to optimize their schedule based on personal preferences, such as avoiding 8 AMs, avoiding certain professors or subjects.

## Project Goal
The goal of this project is to develop an AI-powered advising assistant that takes your current transcript and based on that it can advise you as to what classes
you should take in order to graduate in 4 years. To keep this simple and for the scope of this lass, we will only consider advising for Computer Science majors only.
The system will ingest the official class schedule and degree requirements, interact
with the student to understand their preferences, and generate valid schedule options that ensure timely graduation.

## System Architecture & Tools
  - **Language Model:** Llama 3 (via Ollama) for natural language interaction.
  - **Knowledge Base (RAG):** Vector Database such as Chroma DB
  - **Logic Layer (The "Tool"):** A Python-based constraint checker. The LLM will not guess the schedule; it will query this Python module to verify if a schedule is valid (checking prerequisites and time conflicts).
  - **Interface:** A simple web UI (using Streamlit or Gradio) for chatting and displaying the schedule table.

## Week by Week Plan
  - **Week 1 (Jan 19):** Project Kickoff. Data entry: Create the structured dataset (JSON) of CS classes, prerequisites, and times for the current year.

  - **Week 2:** Build the "Logic Core" (Python functions). Write code that takes a list of taken classes and returns a list of valid next classes.

  - **Week 3:** Setup Ollama and Python API. create the prompt that allows the AI to ask the user for their transcript and preferences.

  - **Week 4:** Integration. Connect the LLM to the Logic Core. The LLM should be able to receive a user request ("I hate 8 AMs") and filter the valid class list accordingly.

  - **Week 5:** Interface Development. Build a basic UI where users can chat and see a visual table of their proposed schedule.

  - **Week 6:** Refinement. Add "Degree Audit" feature—comparing the user's history against graduation requirements.

  - **Week 7:** Testing & Guardrails. Ensure the AI refuses to suggest impossible schedules (e.g., overlapping classes).

  - **Week 8:** Final Polish & Report. Prepare the demo and write the final report on system limitations.