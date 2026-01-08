---
marp: true
theme: default
paginate: true
---

# Artificial Intelligence: A Modern Approach
## Chapter 1: Introduction

Russell & Norvig

---

## Chapter Overview

1. What Is AI?
2. The Foundations of Artificial Intelligence
3. The History of Artificial Intelligence
4. The State of the Art
5. Risks and Benefits of AI

---

# 1.1 What Is AI?

---

## Defining AI: Four Approaches

AI can be defined along two dimensions:

- **Thought vs. Behavior**: Internal reasoning vs. external actions
- **Human-like vs. Rational**: Mimicking humans vs. ideal performance

This creates four categories of AI definitions.

---

## The Four Definitions of AI

| | **Human-like** | **Rational** |
|---|---|---|
| **Thinking** | Thinking Humanly | Thinking Rationally |
| **Acting** | Acting Humanly | Acting Rationally |

---

## Acting Humanly: The Turing Test

**The Turing Test** (Alan Turing, 1950)
- Computer converses with human interrogator
- If interrogator cannot distinguish machine from human, the machine "thinks"

**Capabilities needed:**
- Natural language processing
- Knowledge representation
- Automated reasoning
- Machine learning

---

## Acting Humanly: Total Turing Test

Extends basic Turing Test to include:
- **Computer vision** (to perceive objects)
- **Robotics** (to manipulate objects and move)

These six disciplines comprise much of AI research.

---

## Thinking Humanly: Cognitive Modeling

**Goal**: Make machines think like humans

**Approaches:**
- Introspection: catching our own thoughts
- Psychological experiments: observing humans in action
- Brain imaging: observing the brain in action

**Cognitive Science**: Interdisciplinary field combining AI models with experimental psychology

---

## Thinking Rationally: Laws of Thought

**Aristotelian tradition**: Correct reasoning through syllogisms

**Example:**
- Socrates is a man
- All men are mortal
- Therefore, Socrates is mortal

**Challenges:**
- Informal knowledge is hard to formalize
- Solving problems "in principle" vs. "in practice"
- Computational complexity

---

## Acting Rationally: The Rational Agent

**Rational Agent**: Acts to achieve the best expected outcome

**Advantages over other approaches:**
- More general than "laws of thought"
- Amenable to scientific development
- Handles uncertainty and incomplete information

**Note**: Perfect rationality is not always achievable due to computational limitations

---

## Limited Rationality

- **Bounded rationality**: Acting appropriately when there isn't time to do all computations
- Modern AI focus: building agents that do well given computational resources
- More robust than requiring perfect rationality

---

# 1.2 The Foundations of AI

---

## Foundations Overview

AI draws from multiple disciplines:

1. Philosophy
2. Mathematics
3. Economics
4. Neuroscience
5. Psychology
6. Computer Engineering
7. Control Theory and Cybernetics
8. Linguistics

---

## Philosophy

**Key Questions:**
- Can formal rules derive valid conclusions?
- How does mental mind arise from physical brain?
- Where does knowledge come from?
- How does knowledge lead to action?

**Key Contributors:**
- Aristotle: Logic, foundation of reasoning
- Descartes: Dualism, mind-body problem
- Bacon, Locke: Empiricism (knowledge from experience)

---

## Mathematics

**Key Contributions:**

**Logic and Computation:**
- Boole, Frege: Formal logic
- Gödel: Incompleteness theorem
- Turing: Computability, Church-Turing thesis

**Algorithms:**
- What can be computed?
- What is tractable vs. intractable?
- NP-completeness (Cook, Karp)

---

## Mathematics (continued)

**Probability:**
- Foundation for reasoning under uncertainty
- Bayes' rule (1763)
- Modern probabilistic approaches in AI

**Key Insight**: Distinguishing between decidability, computability, and tractability

---

## Economics

**Key Concepts:**
- **Decision theory**: Making decisions to maximize utility
- **Game theory**: Multi-agent decision making
- **Operations research**: Optimizing scarce resources

**Utility theory**: Formalizing preferences and rational behavior

**Satisficing**: Finding "good enough" solutions when optimization is intractable

---

## Neuroscience

**How do brains process information?**

**Key Discoveries:**
- Neurons as fundamental units
- Synaptic connections and learning
- Neural network structure

**Impact on AI:**
- Artificial neural networks
- Deep learning architectures
- Inspiration for learning algorithms

---

## Computer Engineering

**Hardware enables AI:**

**Evolution:**
- 1940s: First electronic computers
- Moore's Law: Exponential growth in computing power
- Modern GPUs: Parallel processing for neural networks

**Impact**: Increased computational power makes previously intractable problems solvable

---

## Linguistics

**How does language relate to thought?**

**Key Developments:**
- Chomsky: Syntactic structures (1957)
- Computational linguistics
- Natural language understanding

**Knowledge representation**: How language expresses and conveys knowledge

---

# 1.3 The History of AI

---

## The Gestation of AI (1943-1955)

**McCulloch & Pitts (1943)**
- Model of artificial neurons
- Boolean circuit model of brain

**Turing's "Computing Machinery and Intelligence" (1950)**
- Turing Test
- Machine learning, genetic algorithms, reinforcement learning concepts

---

## The Birth of AI (1956)

**Dartmouth Workshop (Summer 1956)**
- John McCarthy, Marvin Minsky, Claude Shannon, Nathaniel Rochester
- Term "Artificial Intelligence" coined
- Optimistic predictions

**Early Programs:**
- Logic Theorist (Newell & Simon)
- Checkers player (Samuel)

---

## Early Enthusiasm (1952-1969)

**Notable Achievements:**
- General Problem Solver (Newell & Simon)
- LISP programming language (McCarthy, 1958)
- Microworlds (blocks world)
- Natural language programs (ELIZA, SHRDLU)

**Optimism**: "Machines will be capable... of doing any work a man can do" (Simon, 1957)

---

## A Dose of Reality (1966-1973)

**Limitations Discovered:**
- Intractability of many problems
- Limited scope of early systems
- Perceptron limitations (Minsky & Papert, 1969)

**AI Winter Begins:**
- Reduced funding
- Unfulfilled predictions
- Need for fundamental knowledge

---

## Knowledge-Based Systems (1969-1979)

**Key Insight**: Intelligence requires knowledge

**Expert Systems:**
- DENDRAL: Chemical structure analysis
- MYCIN: Medical diagnosis
- Capture expert knowledge in rules

**Impact**: Commercial interest in AI revives

---

## AI Becomes an Industry (1980-present)

**Expert System Boom (1980s)**
- Commercial success
- R1/XCON at Digital Equipment Corporation
- AI industry reaches billions in revenue

**Challenges:**
- Knowledge acquisition bottleneck
- Brittleness of expert systems
- Second AI Winter (late 1980s)

---

## AI Adopts the Scientific Method (1987-present)

**Shift in Methodology:**
- Rigorous experimentation
- Statistical significance testing
- Standardized datasets and benchmarks
- Cross-validation

**Focus**: Building on existing work rather than starting from scratch

---

## Emergence of Intelligent Agents (1995-present)

**Whole-Agent Perspective:**
- Integration of multiple AI techniques
- Situated in environments
- Robotic systems

**Examples:**
- Autonomous vehicles
- Household robots
- Interactive agents

---

## Availability of Very Large Datasets (2001-present)

**Big Data Era:**
- Internet provides massive training data
- Web-scale datasets
- Crowdsourcing (ImageNet, etc.)

**Impact**: Data-driven approaches become dominant

---

## Deep Learning Renaissance (2011-present)

**Breakthroughs:**
- Computer vision (ImageNet 2012)
- Speech recognition
- Machine translation
- Game playing (AlphaGo, 2016)

**Key Factors:**
- Big data
- GPU computing
- Improved architectures

---

# 1.4 The State of the Art

---

## What Can AI Do Today?

We'll examine AI capabilities across various domains:

- Autonomous vehicles
- Speech recognition
- Machine translation
- Image understanding
- Game playing
- And more...

---

## Robotic Vehicles

**Achievements:**
- DARPA Grand Challenge (2005): Autonomous desert driving
- Urban Challenge (2007): City traffic navigation
- Modern self-driving cars (Waymo, Tesla, etc.)

**Capabilities:**
- Sensor fusion (cameras, LIDAR, radar)
- Real-time decision making
- Path planning

---

## Speech Recognition

**Progress:**
- Near-human performance on clean audio
- Virtual assistants (Siri, Alexa, Google Assistant)
- Real-time transcription

**Technologies:**
- Deep neural networks
- Recurrent neural networks (RNNs)
- Attention mechanisms

---

## Autonomous Planning and Scheduling

**Applications:**
- NASA Deep Space 1 mission
- Military logistics
- Manufacturing scheduling

**Techniques:**
- Constraint satisfaction
- Search algorithms
- Optimization

---

## Game Playing

**Milestones:**
- Chess: Deep Blue defeats Kasparov (1997)
- Jeopardy!: Watson wins (2011)
- Go: AlphaGo defeats Lee Sedol (2016)
- Poker, StarCraft II, Dota 2

**Significance**: Games as testbeds for AI techniques

---

## Spam Filtering

**Success Story:**
- Filters 99%+ of spam effectively
- Adaptive learning
- Minimal false positives

**Techniques:**
- Machine learning classifiers
- Bayesian filtering
- Neural networks

---

## Logistics and Transportation

**Applications:**
- Route optimization
- Supply chain management
- Fleet management

**Impact**: Billions saved in operational costs

---

## Machine Translation

**Evolution:**
- Rule-based → Statistical → Neural
- Near-human quality for many language pairs
- Real-time translation

**Technologies:**
- Sequence-to-sequence models
- Transformer architecture
- Attention mechanisms

---

## Image Understanding

**Capabilities:**
- Object detection and recognition
- Face recognition
- Medical image analysis
- Image captioning

**Superhuman performance** in some narrow tasks

---

## Medical Diagnosis

**Applications:**
- Radiology analysis
- Disease prediction
- Drug discovery
- Personalized treatment

**Status**: Matching or exceeding human experts in specific tasks

---

# 1.5 Risks and Benefits of AI

---

## The Promise of AI

**Potential Benefits:**
- Economic growth and productivity
- Scientific discoveries
- Improved healthcare
- Enhanced education
- Climate change mitigation
- Reduced human suffering

---

## Lethal Autonomous Weapons

**Concerns:**
- Weapons that select and engage targets without human control
- Accountability questions
- Arms race dynamics
- Lowered barrier to conflict

**Debate**: Ban vs. regulation

---

## Surveillance and Persuasion

**Privacy Concerns:**
- Mass surveillance capabilities
- Data collection and profiling
- Behavior prediction and manipulation

**Examples:**
- Government surveillance
- Targeted advertising
- Social media algorithms

---

## Biased Decision Making

**Algorithmic Bias:**
- Training data reflects historical biases
- Disparate impact on protected groups
- Criminal justice, lending, hiring

**Challenge**: Ensuring fairness while maintaining accuracy

---

## Impact on Employment

**Job Displacement:**
- Automation of routine tasks
- Both blue-collar and white-collar jobs affected
- Pace of change may outstrip adaptation

**Counterarguments:**
- New jobs created
- Historical precedents
- Augmentation vs. replacement

---

## Safety-Critical Applications

**Risks:**
- Self-driving cars
- Medical AI
- Autonomous systems
- Infrastructure control

**Challenge**: Ensuring reliability and safety

---

## Cybersecurity

**AI as Threat:**
- Sophisticated phishing
- Deepfakes
- Automated hacking
- Adversarial attacks on AI systems

**AI as Defense:**
- Threat detection
- Anomaly detection
- Automated response

---

## Long-term Risks: Superintelligence

**Concern**: AI systems that exceed human intelligence

**Questions:**
- Can we ensure alignment with human values?
- Control problem
- Existential risk?

**Debate**: Timeline and likelihood
**Sci-Fi Movies** I-Robot, Avengers: Age of Ultron, Tron: Ares, etc.

---

## Addressing AI Risks

**Approaches:**
- Technical: Robust AI, interpretability, verification
- Policy: Regulation, governance, international cooperation
- Ethics: Value alignment, fairness, accountability
- Education: Preparing workforce, public understanding


---

## Key Takeaways

1. AI is a broad field with multiple definitions and approaches
2. Modern AI emphasizes rational agents and learning from data
3. Dramatic progress in recent years, but general AI remains elusive
4. Important to develop AI responsibly with attention to risks
5. Interdisciplinary nature requires diverse perspectives

---

# Thank You

## Next: Chapter 2 - Intelligent Agents