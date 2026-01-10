# AI Chapter 2: Intelligent Agents Lab
**Russell & Norvig - Artificial Intelligence: A Modern Approach**

---

## Lab Overview

This lab provides hands-on experience with the fundamental concepts of various types of intelligent agents and environments through a series of runnable Python examples. Each exercise demonstrates key ideas from Chapter 2 using the classic vacuum cleaner world as our domain.

**Setup:** Ensure you have Python 3.10+ installed. No external libraries required.

This lab is designed to be run in a Jupyter notebook environment, because the examples build progressively.

I suggest running the following commands from your base user directory:


```bash
mkdir cs430 
cd cs430 
uv init 
uv sync
source .venv/bin/activate
touch agent.ipynb
```

The last command will create a file such as `agent.ipynb`. 

Select the virtual environment created by ollama (`cs430`) as the kernel for your Jupyter notebook.

Paste the code for each exercise in a new code cell.

Make sure to record your answers to *all* reflections.

---

## Exercise 1: Understanding Agents and Environments

### Description
This exercise introduces the basic agent-environment interaction loop. We'll implement a simple two-location vacuum world where an agent can perceive its location and the cleanliness status, then take actions to move or clean.

### Key Concepts
- **Agent**: Perceives environment through sensors, acts through actuators
- **Percept**: Current sensory input `[Location, Status]`
- **Environment**: The world the agent operates in
- **Performance Measure**: How we evaluate agent success

### Task
**Run the code below** and observe:
- How percepts are structured as `(Location, Status)` tuples
- The actions available to the agent
- How the environment state changes after each action
- The performance score calculation

```python
"""
Exercise 1: Agents and Environments
"""
import random
from enum import Enum
from typing import Tuple

class Location(Enum):
    A = "A"
    B = "B"

class Status(Enum):
    CLEAN = "Clean"
    DIRTY = "Dirty"

class Action(Enum):
    LEFT = "Left"
    RIGHT = "Right"
    SUCK = "Suck"
    NOOP = "NoOp"

class VacuumEnvironment:
    """Simple two-location vacuum environment"""
    def __init__(self):
        self.locations = {Location.A: Status.DIRTY, Location.B: Status.DIRTY}
        self.agent_location = Location.A
        self.performance = 0
        self.time_steps = 0
    
    def percept(self) -> Tuple[Location, Status]:
        """Return current percept: [Location, Status]"""
        return (self.agent_location, self.locations[self.agent_location])
    
    def execute(self, action: Action):
        """Execute action and update environment"""
        self.time_steps += 1
        
        if action == Action.SUCK:
            if self.locations[self.agent_location] == Status.DIRTY:
                self.locations[self.agent_location] = Status.CLEAN
                self.performance += 10  # Reward for cleaning
        elif action == Action.LEFT:
            self.agent_location = Location.A
            self.performance -= 1  # Cost of movement
        elif action == Action.RIGHT:
            self.agent_location = Location.B
            self.performance -= 1  # Cost of movement
    
    def is_clean(self) -> bool:
        return all(status == Status.CLEAN for status in self.locations.values())
    
    def __str__(self):
        return f"[{self.locations[Location.A].value}] Agent@{self.agent_location.value} [{self.locations[Location.B].value}] | Perf: {self.performance}"

# Demonstration
print("="*70)
print("EXERCISE 1: Agent-Environment Interaction")
print("="*70)

env = VacuumEnvironment()
print(f"\nInitial state: {env}")
print(f"Current percept: {env.percept()}")
print("\nExecuting actions manually:")

# Manual action sequence
env.execute(Action.SUCK)
print(f"After SUCK: {env}")

env.execute(Action.RIGHT)
print(f"After RIGHT: {env}")

env.execute(Action.SUCK)
print(f"After SUCK: {env}")

print(f"\nFinal Performance: {env.performance}")
```

### Reflection Questions
1. **What is the difference between the agent's percept and the full environment state?** Consider what information is hidden from the agent.
2. **How does the performance measure influence what actions are "good"?** What would/could happen if we changed the reward/cost values?

---

## Exercise 2: Rationality and Agent Functions

### Description
This exercise demonstrates a **simple reflex agent** that maps percepts directly to actions using condition-action rules. This is the simplest agent architecture and illustrates the concept of rational behavior.

### Key Concepts
- **Agent Function**: Abstract mapping from percept sequences to actions
- **Agent Program**: Concrete implementation that runs on architecture
- **Rationality**: Maximizing expected performance given percept sequence and knowledge
- **Reflex Actions**: Decisions based only on current percept

### Task
**Run the code** and focus on:
- The `simple_reflex_vacuum_agent()` function—this is the agent program
- How the agent decides actions based only on the current percept
- The performance score achieved by this simple strategy
- Whether the agent successfully completes its task

```python
"""
Exercise 2: Simple Reflex Agent (Rational Behavior)
"""

def simple_reflex_vacuum_agent(percept: Tuple[Location, Status]) -> Action:
    """
    Agent function: maps current percept to action
    Rules:
      - If current location is dirty → SUCK
      - If at location A and clean → move RIGHT
      - If at location B and clean → move LEFT
    """
    location, status = percept
    
    if status == Status.DIRTY:
        return Action.SUCK
    elif location == Location.A:
        return Action.RIGHT
    else:
        return Action.LEFT

print("\n" + "="*70)
print("EXERCISE 2: Simple Reflex Agent Behavior")
print("="*70)
print("\nAgent Rules:")
print("  - IF dirty THEN suck")
print("  - IF at A and clean THEN move right")
print("  - IF at B and clean THEN move left")
print()

env = VacuumEnvironment()
for step in range(8):
    percept = env.percept()
    action = simple_reflex_vacuum_agent(percept)
    print(f"Step {step}: {env}")
    print(f"  Percept: {percept} → Action: {action.value}")
    env.execute(action)
    if env.is_clean():
        print(f"\nStep {step+1}: {env}")
        print("✓ All locations clean!")
        break

print(f"\nFinal Performance Score: {env.performance}")
print(f"Steps taken: {env.time_steps}")
```

### Reflection Questions
3. **In this environment, does the agent need memory to act rationally?** Why or why not?
4. **Is this agent rational?** Does it maximize expected performance given its percept sequence?
5. **What problem would this agent encounter in a larger environment (e.g., 10 locations)?** Think about its rule structure.
6. **Could the `simple_reflex_vacuum_agent` get stuck in an infinite loop?** Under what circumstances?

---

## Exercise 3: Environment Properties - Stochastic vs Deterministic

### Description
Real-world environments are rarely deterministic. This exercise shows how **uncertainty** in action outcomes affects agent behavior. We compare the same agent in deterministic vs. stochastic environments.

### Key Concepts
- **Deterministic Environment**: Next state completely determined by current state and action
- **Stochastic Environment**: Actions have uncertain outcomes
- **Robustness**: How well agents handle uncertainty

### Task
**Run the code** and observe:
- How the stochastic environment causes SUCK actions to sometimes fail
- The impact on performance and number of steps required
- Whether the agent adapts to failures (spoiler: it doesn't, because it's memoryless)

```python
"""
Exercise 3: Environment Properties - Stochasticity
"""

class StochasticVacuumEnvironment(VacuumEnvironment):
    """Vacuum environment where SUCK action may fail"""
    def execute(self, action: Action):
        self.time_steps += 1
        
        if action == Action.SUCK:
            if self.locations[self.agent_location] == Status.DIRTY:
                # Only 70% success rate
                if random.random() > 0.3:
                    self.locations[self.agent_location] = Status.CLEAN
                    self.performance += 10
                else:
                    print("    ⚠ SUCK action failed!")
        elif action == Action.LEFT:
            self.agent_location = Location.A
            self.performance -= 1
        elif action == Action.RIGHT:
            self.agent_location = Location.B
            self.performance -= 1

print("\n" + "="*70)
print("EXERCISE 3: Stochastic Environment")
print("="*70)
print("\nEnvironment: SUCK action has 70% success rate")
print()

random.seed(42)  # For reproducible results
env_stochastic = StochasticVacuumEnvironment()

for step in range(15):
    percept = env_stochastic.percept()
    action = simple_reflex_vacuum_agent(percept)
    print(f"Step {step}: {env_stochastic}")
    print(f"  Action: {action.value}")
    env_stochastic.execute(action)
    if env_stochastic.is_clean():
        print(f"\n✓ All locations clean!")
        break

print(f"\nFinal Performance: {env_stochastic.performance}")
print(f"Total steps: {env_stochastic.time_steps}")
```

### Reflection Questions
7. **Does the simple reflex agent behave rationally in the stochastic environment?** Why or why not?
8. **What additional capability would help the agent handle stochasticity better?** Think about typical failure logic.
9. **Compare the performance scores:** How much worse is performance in the stochastic environment?

---

## Exercise 4: Model-Based Reflex Agents

### Description
When environments are **partially observable** or when past actions matter, agents need **internal state**. This exercise shows a model-based agent that tracks what it knows about the world.

### Key Concepts
- **Internal State**: Agent's memory of aspects of the world
- **Model**: Knowledge of how the world evolves
- **State Update**: Incorporating new percepts into internal representation

### Task
**Run the code** and pay attention to:
- The `model` dictionary that tracks believed status of both locations
- How the model gets updated with each percept
- The agent's decision-making based on its model, not just current percept
- When the agent decides it's done (NOOP action)

```python
"""
Exercise 4: Model-Based Reflex Agent
"""

class ModelBasedVacuumAgent:
    """Agent that maintains internal state about the world"""
    def __init__(self):
        # Internal model of the world state
        self.model = {Location.A: Status.DIRTY, Location.B: Status.DIRTY}
        self.location = Location.A
    
    def agent_program(self, percept: Tuple[Location, Status]) -> Action:
        location, status = percept
        
        # Update internal model based on percept
        self.location = location
        self.model[location] = status
        
        # Decide action based on model
        if status == Status.DIRTY:
            return Action.SUCK
        elif self.model[Location.A] == Status.DIRTY:
            return Action.LEFT
        elif self.model[Location.B] == Status.DIRTY:
            return Action.RIGHT
        else:
            return Action.NOOP  # Believes all locations clean

print("\n" + "="*70)
print("EXERCISE 4: Model-Based Reflex Agent")
print("="*70)
print("\nAgent maintains internal model of both locations")
print()

env = VacuumEnvironment()
agent = ModelBasedVacuumAgent()

for step in range(10):
    percept = env.percept()
    action = agent.agent_program(percept)
    print(f"Step {step}: {env}")
    print(f"  Agent's Model: A={agent.model[Location.A].value}, B={agent.model[Location.B].value}")
    print(f"  Action: {action.value}")
    env.execute(action)
    print()
    if action == Action.NOOP:
        print("✓ Agent believes task is complete")
        break

print(f"Final Performance: {env.performance}")
```

### Reflection Questions
10. **How is the model-based agent different from the simple reflex agent?** What additional capability does it have?
11. **Could this agent handle a partially observable environment** (will it get stuck infinitely)? Why?
12. **What would happen if the environment changed while the agent wasn't looking** (e.g., location A gets dirty again)? Would the agent notice?

---

## Exercise 5: Goal-Based Agents

### Description
**Goal-based agents** explicitly represent desirable states and plan to achieve them. This makes them more flexible than reflex agents, as the same goal can be achieved through different paths.

### Key Concepts
- **Goals**: Explicit representation of desirable situations
- **Planning**: Considering future consequences of actions
- **Flexibility**: Multiple paths to the same goal

### Task
**Run the code** and notice:
- The explicit goal: "visit and clean all locations"
- The `visited` set that tracks progress toward the goal
- How the agent plans its actions to achieve the goal
- The difference between goals (what to achieve) and actions (how to achieve it)

```python
"""
Exercise 5: Goal-Based Agent
"""

class GoalBasedVacuumAgent:
    """Agent that plans actions to achieve explicit goals"""
    def __init__(self):
        self.goal = "all locations visited and clean"
        self.visited = set()
        self.cleaned = set()
    
    def agent_program(self, percept: Tuple[Location, Status]) -> Action:
        location, status = percept
        self.visited.add(location)
        
        # Subgoal: Clean current location if dirty
        if status == Status.DIRTY:
            return Action.SUCK
        else:
            self.cleaned.add(location)
        
        # Plan: Visit unvisited locations
        if Location.A not in self.visited:
            return random.choice([Action.LEFT, Action.RIGHT])
        elif Location.B not in self.visited:
            return random.choice([Action.LEFT, Action.RIGHT])
        else:
            # Goal achieved: all locations visited and clean
            return Action.NOOP

print("\n" + "="*70)
print("EXERCISE 5: Goal-Based Agent")
print("="*70)
print("\nGoal: Visit and clean all locations")
print()

env = VacuumEnvironment()
agent = GoalBasedVacuumAgent()

for step in range(10):
    percept = env.percept()
    action = agent.agent_program(percept)
    print(f"Step {step}: {env}")
    print(f"  Visited: {sorted([loc.value for loc in agent.visited])}")
    print(f"  Cleaned: {sorted([loc.value for loc in agent.cleaned])}")
    print(f"  Action: {action.value}")
    env.execute(action)
    print()
    if action == Action.NOOP:
        print("✓ Goal achieved!")
        break

print(f"Final Performance: {env.performance}")
```

### Reflection Questions
13. **How does having an explicit goal make the agent more flexible?** Consider alternative scenarios (e.g., blocked paths).
14. **What's the difference between the goal and the plan?** Why is this separation useful? Consider a different grid, for example.
15. **Could this agent adapt if the goal changed mid-execution** (e.g., "now clean location A twice")? What would need to change?

---

## Exercise 6: Utility-Based Agents

### Description
**Utility-based agents** handle trade-offs by assigning numerical values (utilities) to states. They choose actions that maximize expected utility, allowing them to balance competing objectives.

### Key Concepts
- **Utility Function**: Maps states to real numbers representing "happiness"
- **Expected Utility**: Predicted utility considering action outcomes
- **Trade-offs**: Balancing multiple objectives (speed vs. thoroughness)

### Task
**Run the code** and observe:
- The `utility()` function that scores states
- The `expected_utility()` function that predicts outcomes
- How the agent chooses actions by comparing expected utilities
- The trade-off between cleaning (high utility) and moving (cost)

```python
"""
Exercise 6: Utility-Based Agent
"""

class UtilityBasedVacuumAgent:
    """Agent that maximizes expected utility"""
    def __init__(self):
        self.state = {Location.A: Status.DIRTY, Location.B: Status.DIRTY}
        self.location = Location.A
    
    def utility(self, state: dict) -> float:
        """Calculate utility of a state"""
        clean_count = sum(1 for s in state.values() if s == Status.CLEAN)
        return clean_count * 10  # 10 points per clean location
    
    def expected_utility(self, action: Action) -> float:
        """Predict utility after taking action (minus costs)"""
        next_state = self.state.copy()
        next_location = self.location
        
        if action == Action.SUCK:
            if self.state[self.location] == Status.DIRTY:
                next_state[self.location] = Status.CLEAN
            return self.utility(next_state) - 0  # No cost
        elif action == Action.LEFT:
            next_location = Location.A
            # Expected utility: might find dirt there
            if next_state[Location.A] == Status.DIRTY:
                # If we know it's dirty, we can clean it (net +9)
                return self.utility(next_state) + 10 - 1
            return self.utility(next_state) - 1  # Just movement cost
        elif action == Action.RIGHT:
            next_location = Location.B
            # Expected utility: might find dirt there
            if next_state[Location.B] == Status.DIRTY:
                # If we know it's dirty, we can clean it (net +9)
                return self.utility(next_state) + 10 - 1
            return self.utility(next_state) - 1  # Just movement cost
        return self.utility(next_state)
    
    def agent_program(self, percept: Tuple[Location, Status]) -> Action:
        location, status = percept
        self.location = location
        self.state[location] = status
        
        # Evaluate all possible actions
        actions = [Action.SUCK, Action.LEFT, Action.RIGHT]
        action_utilities = []
        
        for a in actions:
            eu = self.expected_utility(a)
            action_utilities.append((a, eu))
        
        # Choose action with maximum expected utility
        best_action = max(action_utilities, key=lambda x: x[1])
        
        return best_action[0]

print("\n" + "="*70)
print("EXERCISE 6: Utility-Based Agent")
print("="*70)
print("\nUtility Function: +10 per clean location, -1 per move")
print()

env = VacuumEnvironment()
agent = UtilityBasedVacuumAgent()

for step in range(10):
    percept = env.percept()
    
    # Show utility calculation
    print(f"Step {step}: {env}")
    print(f"  Current state utility: {agent.utility(agent.state)}")
    
    # Get action
    action = agent.agent_program(percept)
    print(f"  Chosen action: {action.value}")
    
    env.execute(action)
    print()
    if env.is_clean():
        print("✓ All locations clean!")
        break

print(f"Final Performance: {env.performance}")
```

### Reflection Questions
16. **How does the utility function encode the agent's preferences?** What would happen if we changed the values?
17. **When would a utility-based agent be better than a goal-based agent?** Think about scenarios with competing objectives.
18. **How could we extend the utility function** to consider time (e.g., finish faster gets higher utility)?

---

## Exercise 7: Comparing All Agent Types

### Description
This final exercise runs all four agent types in the same environment and compares their performance. This illustrates the trade-offs between simplicity and capability.

### Key Concepts
- **Performance Comparison**: Quantitative evaluation of different architectures
- **Complexity vs. Capability**: More sophisticated agents may achieve better performance
- **Architecture Selection**: Matching agent type to environment properties

### Task
**Run the code** and analyze:
- Which agent types achieve the best performance
- Why some agents perform better than others
- The relationship between agent complexity and performance
- When simpler agents are "good enough"

```python
"""
Exercise 7: Comparing All Agent Types
"""

def run_agent_comparison():
    """Run all four agent types and compare performance"""
    
    def run_simple_reflex(max_steps=10):
        env = VacuumEnvironment()
        for _ in range(max_steps):
            action = simple_reflex_vacuum_agent(env.percept())
            env.execute(action)
            if env.is_clean():
                break
        return env.performance, env.time_steps
    
    def run_model_based(max_steps=10):
        env = VacuumEnvironment()
        agent = ModelBasedVacuumAgent()
        for _ in range(max_steps):
            action = agent.agent_program(env.percept())
            env.execute(action)
            if action == Action.NOOP:
                break
        return env.performance, env.time_steps
    
    def run_goal_based(max_steps=10):
        env = VacuumEnvironment()
        agent = GoalBasedVacuumAgent()
        for _ in range(max_steps):
            action = agent.agent_program(env.percept())
            env.execute(action)
            if action == Action.NOOP:
                break
        return env.performance, env.time_steps
    
    def run_utility_based(max_steps=10):
        env = VacuumEnvironment()
        agent = UtilityBasedVacuumAgent()
        for _ in range(max_steps):
            action = agent.agent_program(env.percept())
            env.execute(action)
            if env.is_clean():
                break
        return env.performance, env.time_steps
    
    return {
        "Simple Reflex": run_simple_reflex(),
        "Model-Based": run_model_based(),
        "Goal-Based": run_goal_based(),
        "Utility-Based": run_utility_based()
    }

print("\n" + "="*70)
print("EXERCISE 7: Agent Performance Comparison")
print("="*70)
print("\nRunning all four agent types in identical environments...\n")

results = run_agent_comparison()

print("Results:")
print("-" * 70)
print(f"{'Agent Type':<20} {'Performance':<15} {'Steps':<10}")
print("-" * 70)
for agent_type, (perf, steps) in results.items():
    print(f"{agent_type:<20} {perf:<15} {steps:<10}")
print("-" * 70)

# Find best performer
best_agent = max(results.items(), key=lambda x: x[1][0])
print(f"\n Best Performance: {best_agent[0]} (Score: {best_agent[1][0]})")
```

### Reflection Questions
19. **Did all agents achieve the same performance?** If not, why do you think they differ?
20. **In general, is the highest-performing agent always the "best" choice?** Consider factors beyond performance score.
21. **For this specific environment (small, deterministic, fully observable), is the complexity of utility-based agents justified?** When would it be justified?

---

## Submission Instructions

Create a new **public** Github Repository called `cs430`, upload your local `cs430` folder there including the `agent.ipynb` file from this lab and:

A Markdown document called `reflections.md` containing this header

Create `lab7_results.md`:

```markdown
# Names: Your names here
# Lab: lab1 (Intelligent Agents)
# Date: Today's date
```

And your answers to all 21 reflection questions above. Each answer should be 2-5 sentences that demonstrate your understanding of the concepts through the lens of the exercises you ran.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the first lab!

---

## Optional Extensions

If you want to explore further, try modifying the code to:

1. **Add more locations** to the vacuum world
2. **Implement a dynamic environment** that re-dirties locations randomly
3. **Create a performance measure** that penalizes time more heavily

---

**Lab Complete!** You now understand the foundational concepts of intelligent agents and how environment properties influence agent design.