---
marp: true
theme: default
paginate: true
---

# Artificial Intelligence: A Modern Approach
## Chapter 2: Intelligent Agents

Russell & Norvig

---

## Chapter Overview

1. Agents and Environments
2. Good Behavior: The Concept of Rationality
3. The Nature of Environments
4. The Structure of Agents

---

## Introduction

**Central Question**: What is an intelligent agent?

**Key Idea**: An agent is anything that perceives its environment through sensors and acts upon that environment through actuators.

This chapter develops the foundational concepts for understanding intelligent systems.

---

# 2.1 Agents and Environments

---

## What Is an Agent?

**Agent**: Anything that can be viewed as:
- **Perceiving** its environment through **sensors**
- **Acting** upon that environment through **actuators**

**Percept**: The agent's perceptual inputs at any given instant

**Percept sequence**: Complete history of everything the agent has perceived

---

## Agent Function vs. Agent Program

**Agent Function**: Abstract mathematical description
- Maps any given percept sequence to an action
- `f: P* → A`

**Agent Program**: Concrete implementation
- Runs on a physical architecture
- Implements the agent function

**Goal**: Design the best agent program for given architecture

---

## Example: Vacuum Cleaner Agent

**Environment**: Two locations (A and B)
- Each location can be clean or dirty
- Agent can perceive its location and whether it's dirty

**Actions**:
- `Left`: Move left
- `Right`: Move right
- `Suck`: Clean current location
- `NoOp`: Do nothing

---

## Vacuum World Percepts

**Percept**: `[Location, Status]`

Examples:
- `[A, Dirty]`
- `[B, Clean]`

**Percept Sequence Example**:
```
[A, Dirty], [A, Clean], [B, Dirty]
```

---

## Simple Vacuum Agent Function

| Percept Sequence | Action |
|-----------------|--------|
| [A, Clean] | Right |
| [A, Dirty] | Suck |
| [B, Clean] | Left |
| [B, Dirty] | Suck |

**Note**: This is a partial specification (only looks at current percept)

---

## Agents in Different Domains

**Human Agent**:
- Sensors: Eyes, ears, skin, etc.
- Actuators: Hands, legs, vocal tract, etc.

**Robotic Agent**:
- Sensors: Cameras, infrared, sonar
- Actuators: Motors, grippers

**Software Agent**:
- Sensors: Keyboard input, file contents, network packets
- Actuators: Display output, file operations, network messages

---

## The Agent-Environment Interaction

```
     ┌───────┐
     │ Agent │
     └───┬───┘
         │
    ┌────┴────┐
    │         │
Sensors   Actuators
    │         │
    └────┬────┘
         │
   ┌─────▼─────┐
   │Environment│
   └───────────┘
```

Continuous cycle: percepts → agent → actions → environment

---

# 2.2 Good Behavior: The Concept of Rationality

---

## Performance Measures

**Key Question**: How do we evaluate agent behavior?

**Performance Measure**: Objective criterion for success
- Evaluates environment state sequence
- Defined by designer, not agent
- Should measure what we actually want

**Example (Vacuum)**: Amount of dirt cleaned? Clean squares over time? Energy efficiency?

---

## What Is Rationality?

**Rational Agent**: Selects actions that maximize expected performance measure, given:
1. The percept sequence so far
2. Built-in knowledge


---

## Rationality Definition (Formal)

For each possible percept sequence, a rational agent should select an action that is expected to maximize its performance measure, given:

1. The evidence provided by the percept sequence
2. Whatever built-in knowledge the agent has

---

## Rationality Example: Vacuum Agent

**Scenario**: Agent starts in location A

**Performance Measure Options**:
1. +1 per clean square per time step
2. +10 for each square cleaned
3. -1 per action

Different measures lead to different optimal behaviors!

---

## Omniscience vs. Rationality

**Omniscience**: Knows actual outcome of actions
- Impossible in practice
- Not required for rationality

**Rationality**: Makes best decision given available information
- Achievable
- Based on percept sequence and prior knowledge

**Example**: Crossing street rationally but getting hit by falling airplane

---

## Information Gathering

**Rational agents gather information**:
- Exploration vs. exploitation
- Looking before crossing street
- Scientific experiments

**Percepts provide evidence** about environment state

Doing actions to modify future percepts is essential to rationality.

---

## Learning and Autonomy

**Learning**: Modifying behavior based on experience
- Essential for true rationality
- Compensates for incomplete initial knowledge

**Autonomy**: Extent to which behavior is determined by experience vs. prior knowledge
- High autonomy: learns from environment
- Low autonomy: relies on built-in knowledge

**Best agents**: Combine prior knowledge with learning

---

## The Rationality Definition Refined

A rational agent should:
1. **Gather information** through percepts
2. **Learn** from experience
3. **Perform actions** to maximize expected performance
4. Be as **autonomous** as possible

---

# 2.3 The Nature of Environments

---

## Task Environments

**PEAS Description**: 
- **P**erformance measure
- **E**nvironment
- **A**ctuators
- **S**ensors

Task environment = problem specification for intelligent agent

---

## PEAS Example: Automated Taxi

**Performance**: Safe, fast, legal, comfortable, maximize profits

**Environment**: Roads, traffic, pedestrians, weather, customers

**Actuators**: Steering, accelerator, brake, horn, display

**Sensors**: Cameras, GPS, speedometer, odometer, sonar, accelerometer

---

## PEAS Example: Medical Diagnosis

**Performance**: Healthy patient, minimize costs, lawsuits

**Environment**: Patient, hospital, staff

**Actuators**: Screen display (questions, tests, diagnoses, treatments)

**Sensors**: Keyboard (patient symptoms, test results)

---

## PEAS Example: Part-Picking Robot

**Performance**: Percentage correct parts, speed

**Environment**: Conveyor belt, parts, bins

**Actuators**: Jointed arm, gripper

**Sensors**: Camera, joint angle sensors

---

## PEAS Example: Interactive Tutor

**Performance**: Student test scores, engagement

**Environment**: Student, curriculum

**Actuators**: Screen display (exercises, suggestions, corrections)

**Sensors**: Keyboard input, student responses

---

## Properties of Task Environments

We can classify environments along several dimensions:

1. Fully observable vs. Partially observable
2. Single-agent vs. Multi-agent
3. Deterministic vs. Stochastic
4. Episodic vs. Sequential
5. Static vs. Dynamic
6. Discrete vs. Continuous
7. Known vs. Unknown

---

## Observable: Fully vs. Partially

**Fully Observable**:
- Sensors detect all relevant aspects
- Complete state information
- Example: Chess (can see entire board)

**Partially Observable**:
- Incomplete or noisy information
- Must maintain internal state
- Example: Poker (can't see opponent's cards)

---

## Agents: Single vs. Multi-agent

**Single-agent**:
- Agent alone in environment
- Example: Crossword puzzle

**Multi-agent**:
- Other agents present
- Can be competitive or cooperative
- Example: Chess (competitive), self-driving cars (cooperative/competitive)

**Communication** may arise in multi-agent environments

---

## Multi-agent Considerations

**Competitive**: Adversarial behavior
- Chess, poker
- Randomized strategies may be optimal

**Cooperative**: Collaboration
- Taxi avoidance
- Communication emerges

**Mixed**: Both competitive and cooperative elements

---

## Deterministic vs. Stochastic

**Deterministic**:
- Next state completely determined by current state and action
- Example: Chess

**Stochastic**:
- Uncertainty in outcomes
- Example: Dice games, robot movement

**Note**: Partial observability can appear stochastic (uncertainty about state vs. outcome)

---

## Episodic vs. Sequential

**Episodic**:
- Experience divided into episodes
- Next episode doesn't depend on actions in previous
- Example: Defect inspection, image classification

**Sequential**:
- Current decision affects future decisions
- Must think ahead
- Example: Chess, taxi driving

---

## Static vs. Dynamic

**Static**:
- Environment unchanged while agent deliberates
- Example: Crossword, chess with no clock

**Dynamic**:
- Environment changes during deliberation
- Time matters
- Example: Taxi driving, real-time games

**Semidynamic**: Environment doesn't change, but performance score does (chess with clock)

---

## Discrete vs. Continuous

**Discrete**:
- Finite number of distinct states/actions/percepts
- Example: Chess

**Continuous**:
- Infinite possible values
- Example: Taxi driving (speed, location)

Can apply to:
- State of environment
- Time
- Percepts and actions

---

## Known vs. Unknown

**Known Environment**:
- Agent knows "laws of physics"
- Outcomes of actions known
- May still be partially observable

**Unknown Environment**:
- Must learn how environment works
- Exploration required

**Note**: This is about agent's knowledge, not observability

---

## Hardest vs. Easiest Environments

**Hardest**: 
- Partially observable
- Multi-agent
- Stochastic
- Sequential
- Dynamic
- Continuous
- Unknown

---

**Easiest**:
- Fully observable
- Single-agent
- Deterministic
- Episodic
- Static
- Discrete
- Known

---

## Real-World Complexity

Most real-world environments are:
- Partially observable
- Stochastic
- Sequential
- Dynamic
- Continuous
- Multi-agent

This is why AI is challenging!

---

# 2.4 The Structure of Agents

---

## Agent = Architecture + Program

**Architecture**: Computing device with sensors and actuators
- Physical system
- PC, robotic car, collection of software modules

**Agent Program**: Implementation of agent function
- Maps percepts to actions
- Runs on the architecture

---

## Basic Agent Program Structure

```python
def agent_program(percept):
    # Returns an action
    return action
```

**Note**: Takes current percept only, but must have memory to consider history

---

## Table-Driven Agent

**Idea**: Store agent function as a table
- Lookup percept sequence → action

```python
def table_driven_agent(percept):
    percepts.append(percept)
    action = lookup(percepts, table)
    return action
```

---

## Problems with Table-Driven Agents

**Impractical** for most environments:

1. **Huge tables**: For n percepts over t time steps: n^t entries
2. **Construction time**: No time to build table
3. **No autonomy**: Designer must anticipate everything
4. **No learning**: Fixed behavior

**Example**: Chess has ~10^40 states—table won't fit in universe!

---

## Four Basic Agent Types

1. **Simple Reflex Agents**
2. **Model-Based Reflex Agents**
3. **Goal-Based Agents**
4. **Utility-Based Agents**

Each adds capabilities and complexity.

---

## Simple Reflex Agents

**Key Idea**: Select actions based on current percept only
- Ignore percept history
- Condition-action rules

```python
def simple_reflex_agent(percept):
    state = interpret_input(percept)
    rule = rule_match(state, rules)
    action = rule.action
    return action
```

---

## Simple Reflex Agent Diagram

```
  Percept
     ↓
What the world
  is like now
     ↓
  Condition-Action
     Rules
     ↓
  Action
```

**Example**: "If car-in-front-is-braking then initiate-braking"

---

## When Do Simple Reflex Agents Work?

**Work well when**:
- Environment is fully observable
- Correct decision can be made from current percept

**Example**: Vacuum cleaner in small, observable room

**Fail when**:
- Important information not in current percept
- Randomness needed

---

## Model-Based Reflex Agents

**Problem**: What if environment partially observable?

**Solution**: Maintain internal state
- Track aspects of world not currently visible
- Update state based on percepts and actions

Requires **model** of how world evolves

---

## Model-Based Agent Structure

```python
def model_based_reflex_agent(percept):
    state = update_state(state, action, percept, model)
    rule = rule_match(state, rules)
    action = rule.action
    return action
```

**Two models needed**:
1. How world evolves independently
2. How agent's actions affect world

---

## Model-Based Agent Diagram

```
      Percept
         ↓
   ┌─────────────┐
   │How the world│  State
   │   evolves   │   ↓
   └─────────────┘   │
         ↑           │
      What my    What the
      actions do  world is
         ↑        like now
         │           ↓
      Action ← Condition-Action
                   Rules
```

---

## Goal-Based Agents

**Beyond reflexes**: What if multiple actions satisfy rules?

**Solution**: Add goals
- Represents desirable situations
- Agent chooses actions to achieve goals
- Requires **search** and **planning**

More flexible than reflex agents

---

## Goal-Based Agent Structure

```python
def goal_based_agent(percept):
    state = update_state(state, action, percept, model)
    # Search/plan to achieve goals
    action = plan(state, goals, model)
    return action
```

**Key**: Considers future consequences of actions

---

## Goal-Based Agent Diagram

```
      Percept
         ↓
      State
         ↓
   What it will be
   like if I do
   action A
         ↓
      Goals
         ↓
   What action should
      I do now
         ↓
      Action
```

---

## Goals vs. Reflex Rules

**Advantages of goals**:
- **Flexible**: Same goal, different routes
- **Adaptable**: Goals unchanged when environment changes
- **Understandable**: Clear what agent trying to achieve

**Example**: Taxi routing
- Goal: Reach destination
- Multiple paths possible
- Can adapt to road closures

---

## Utility-Based Agents

**Problem**: Goals alone insufficient
- Multiple ways to achieve goal (which is best?)
- Trade-offs between conflicting goals
- Uncertainty about goal achievement

**Solution**: Utility function
- Maps states to real numbers (degree of happiness)
- Agent maximizes expected utility

---

## Utility-Based Agent Structure

```python
def utility_based_agent(percept):
    state = update_state(state, action, percept, model)
    # Choose action that maximizes expected utility
    action = max_utility(state, actions, utility, model)
    return action
```

---

## Utility-Based Agent Diagram

```
      Percept
         ↓
      State
         ↓
   What it will be
   like if I do
   action A
         ↓
   How happy will
    I be then?
    (Utility)
         ↓
   What action should
      I do now
         ↓
      Action
```

---

## Why Utility Functions?

**Advantages**:
1. **Handle trade-offs**: Speed vs. safety in taxi
2. **Multiple goals**: Can weight importance
3. **Uncertainty**: Expected utility when outcomes probabilistic
4. **Rational decisions**: Formalize preferences

---

## Learning Agents

**Most powerful**: Agents that can improve performance

**Key Idea**: Start with basic capabilities, learn from experience

Enables operation in initially unknown environments

---

## Learning Agent Components

Four conceptual components:

1. **Learning Element**: Makes improvements
2. **Performance Element**: Selects actions (one of the previous agent types)
3. **Critic**: Provides feedback on performance
4. **Problem Generator**: Suggests exploratory actions

---

## Learning Agent Diagram

```
   Environment
      ↓ ↑
   Sensors Actuators
      ↓      ↑
   Percepts Actions
      ↓      ↑
   ┌──────────────────┐
   │                  │
   │ Performance  ←───┤ Learning
   │  Element         │  Element
   │      ↑           │       ↑
   └──────┼───────────┘       │
          │    Performance    │
          │      Standard     │
      Problem        Critic   │
      Generator  →   ↓        │
          └─────→ Changes ────┘
```

---

## Performance Element

**Function**: Select external actions
- This is the agent we've been discussing
- Can be reflex, model-based, goal-based, or utility-based

**Gets improved** by learning element

---

## Learning Element

**Function**: Improve performance element
- Uses critic feedback
- Modifies components to improve future performance

**Decisions**:
- What to modify
- How to modify it

---

## Critic

**Function**: Provide feedback
- Evaluates agent's behavior
- Uses fixed performance standard
- Not the same as performance measure

**Example**: Tells taxi "that was bad braking" vs. counting passengers delivered

**Essential**: Learning requires feedback

---

## Problem Generator

**Function**: Suggest exploratory actions
- May lead to suboptimal short-term performance
- Discover better long-term strategies

**Exploration vs. Exploitation Trade-off**:
- Exploitation: Use current knowledge
- Exploration: Try new actions to learn

---

## Learning in Different Environments

**Fully observable, known**: Easier to learn
- Clear feedback
- Can build accurate models

**Partially observable, unknown**: Harder
- Ambiguous feedback
- Must explore to discover

Most real environments require learning

---

## How Agents Represent State

**Atomic**: State as indivisible unit
- No internal structure
- Example: State A, State B

**Factored**: State as variables
- Example: Position = (x, y), Velocity = v
- Most common in practice

**Structured**: Objects and relationships
- Relational/first-order representations
- Example: Block(A) on Block(B)

---

## Atomic Representations

**Used in**:
- Search algorithms (states as nodes)
- Game playing
- Hidden Markov Models

**Advantage**: Simple

**Disadvantage**: No structure to exploit

---

## Factored Representations

**Used in**:
- Constraint satisfaction problems
- Planning
- Propositional logic
- Neural networks (feature vectors)

**Advantage**: Natural, efficient for many problems

---

## Structured Representations

**Used in**:
- First-order logic
- Knowledge bases
- Relational databases

**Advantage**: Most expressive, handles complex relationships

**Disadvantage**: More complex reasoning


---

## Chapter 2 Summary: Agents

**Agent**: Perceives and acts to maximize performance measure

**Rational Agent**: Chooses actions to maximize expected performance given percepts and knowledge

**Key**: Rationality ≠ omniscience or perfection

---

## Chapter 2 Summary: Environments

**Task Environment (PEAS)**:
- Performance, Environment, Actuators, Sensors

**Environment Properties**:
- Observable, deterministic, episodic, static, discrete

**Complexity**: Real environments often partially observable, stochastic, sequential, dynamic

---

## Chapter 2 Summary: Agent Types

**Four Basic Types**:
1. Simple reflex (current percept)
2. Model-based (internal state)
3. Goal-based (explicit goals)
4. Utility-based (preferences)

**Learning Agents**: Can improve over time

---

## Chapter 2 Summary: Design Principles

**Good agent design**:
- Match architecture to environment
- Include learning when possible!
- Balance exploration and exploitation
- Choose appropriate state representation
- Consider autonomy vs. prior knowledge

---

## Key Takeaways

1. Agents are the central abstraction in AI
2. Rationality is about maximizing expected performance
3. Environment properties determine agent complexity
4. Learning enables adaptation to unknown environments
5. Utility functions formalize preferences and trade-offs

---

## Looking Ahead

**Next chapters** build on agent framework:
- Search (problem-solving agents)
- Logic (knowledge-based agents)
- Planning (goal-based agents)
- Learning (improving performance)
- Decision theory (utility-based agents)

All share the agent perspective!

---

# Thank You

# LAB: https://github.com/abuach/430students 

## Next: Chapter 3 - Solving Problems by Searching