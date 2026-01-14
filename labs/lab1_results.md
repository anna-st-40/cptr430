# Names: Anna Stefaniv Oickle, Jaime Garcia
# Lab: lab1 (Intelligent Agents)
# Date: Jan 14, 2026

1. ## What is the difference between the agent's percept and the full environment state? Consider what information is hidden from the agent.
    The full environment has two squares, and the squares have clean/dirty status. The percept only knows the status of the square it is currently on.


2. ## How does the performance measure influence what actions are "good"? What would/could happen if we changed the reward/cost values?
    If it cleans, it is rewarded. If it moves, it has a slight punishment. If the cost of moving was raised, especially if it was greater than the reward of cleaning, the agent might never move at all.

1. ## In this environment, does the agent need memory to act rationally? Why or why not?
    No. It does not need to know if the previous square was clean or dirty, it only needs to know if the current one is clean or dirty.

2. ## Is this agent rational? Does it maximize expected performance given its percept sequence?
    Yes, it is rational, because its percept sequence is to keep trying to clean both squares until they are clean.
    
3. ## What problem would this agent encounter in a larger environment (e.g., 10 locations)? Think about its rule structure.
    It wouldn't know which location isn't clean until it checks all locations, expending cost of moving.

4. ## Could the simple_reflex_vacuum_agent get stuck in an infinite loop? Under what circumstances?
    Yes, if the locations kept getting dirty while the agent isn't there.

1. ## Does the simple reflex agent behave rationally in the stochastic environment? Why or why not?
    Yes. The only difference here is that the suck action sometimes fails, and the agent just keeps retrying the suck action until it works.

2. ## What additional capability would help the agent handle stochasticity better? Think about typical failure logic.
    A percept to know why it failed, that would determine whether to retry right away or go do something else and come back later

3. ## Compare the performance scores: How much worse is performance in the stochastic environment?
    The performance scores are the same because there is no penalty for retrying, only for moving; and the agent does not move until the suck action succeeds.

1. ## How is the model-based agent different from the simple reflex agent? What additional capability does it have?
    It keeps an internal memory of what it thinks the world looks like
2. ## Could this agent handle a partially observable environment (will it get stuck infinitely)? Why?
    It assumes that once a location is clean, it will not get dirty again. It also assumes all locations it hasn't seen are dirty until proven otherwise. So, it shouldn't get stuck - it should just clean all and then stop.
3. ## What would happen if the environment changed while the agent wasn't looking (e.g., location A gets dirty again)? Would the agent notice?
    No, it would not notice. And, if it was a location that was previously cleaned, the agent would continue assuming it was clean, and would assume all locations are clean at the end if it never visited that location again.

1. ## How does having an explicit goal make the agent more flexible? Consider alternative scenarios (e.g., blocked paths).
    It can reach everywhere without a predefined path. Because there is not just one path, it can take a detour if a path is blocked.
2. ## What's the difference between the goal and the plan? Why is this separation useful? Consider a different grid, for example.
    The goal can stay the same with different configurations of locations, while the plan will change to suit them.
3. ## Could this agent adapt if the goal changed mid-execution (e.g., "now clean location A twice")? What would need to change?
    It could, if the memory capability was also updated (so that the agent would be able to keep track of how many times the location was cleaned)

1. ## How does the utility function encode the agent's preferences? What would happen if we changed the values?
    The agent prefers to have a clean location, since it gets 10 points per location that is clean. If we increased the penalty for moving beyond the reward for cleaning, it would never move, since it always selects the action with the most utility.
2. ## When would a utility-based agent be better than a goal-based agent? Think about scenarios with competing objectives.
    Better when the goal is ambigious, such as maximizing something. Or, when there are multiple goals (e.g. drive to this location but also don't kill anyone), it decides which one is more important.
3. ## How could we extend the utility function to consider time (e.g., finish faster gets higher utility)?
    Give a penalty for all actions, not just the move action

1. ## Did all agents achieve the same performance? If not, why do you think they differ?
    All except the goal-based agent. This was because the goal-based agent kept visiting different locations randomly, not optimizing performance.
2. ## In general, is the highest-performing agent always the "best" choice? Consider factors beyond performance score.
    Not always. In a more complex scenario, you might not want to hard-code best performance, and instead have the agent make its own decisions.
3. ## For this specific environment (small, deterministic, fully observable), is the complexity of utility-based agents justified? When would it be justified?
    Here, the simple reflex and the utility-based agents had the same performance in the same amount of steps, while the simple reflex agent was much simpler to code. So, the utility-based agent was not justified in this scenario. In a larger, non-deterministic, not fully observable environment, we would not be able to accomplish the goal with a simple reflex agent, so the utility-based agent would be justified.