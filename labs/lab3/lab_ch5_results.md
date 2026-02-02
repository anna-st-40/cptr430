**Question 1:** Why is the win rate between two random agents approximately equal, and what does this tell us about the relationship between starting position and strategy in Tron?

Both random agents are using the same strategy, and are getting approximately equal win rates. This tells us that the starting position in this case does not affect the strategy or the win rate.

**Question 2:** Random agents occasionally win by "accidentally" making good moves. Explain why this approach cannot scale to more complex games and what properties an intelligent agent needs that random selection lacks.

It is completely random, with no effective intelligence. It does not take into consideration its environment or its opponents. In a more complex game, this would be entirely useless.

**Question 3:** How does the average game length between random agents compare to what you might expect from intelligent play, and what does this reveal about the relationship between lookahead and survival time?

From intelligent play, I would expect to see most of the board filled up (closer to 70 moves), whereas here we only get about 10-30 moves before one agent dies. The more an agent is able to look ahead, the more we would expect it to be able to survive, since it can know what it will crash into before it does, and avoid those moves.

**Question 3b (Visualization):** After watching the visualized game, describe how observing the spatial patterns of random movement helps you understand why random agents crash quickly. What visual patterns emerge that text-based statistics don't capture?

The agents usually crash into themselves long before they reach their opponent, because they are not keeping track of where their trail is, and they often go in a spiral-like shape because all directions are equally likely to be chosen, instead of exploring more of the area.

**Question 4:** Explain why the greedy agent's flood-fill heuristic is effective against random play, and what assumption about survival it makes that proves generally correct in Tron.

The greedy agent tries to fill up its side of the board, and what ends up happening is it minds its own business while waiting for the random agent to crash into itself. It makes the assumption that controlling more empty space correlates with longer survival, which proves generally correct.

**Question 5:** Describe a scenario where greedy space-maximization could lead to a losing position, demonstrating the difference between local optimality and global strategy.

There are instances where the greedy agent draws itself into a corner, usually in the third column. It creates trails on all sides and then goes into the box of its own making, not realizing it has no way out. Local optimality directs it to go in, while global strategy might predict that this is a dead-end route.

**Question 6:** How does the computational cost of flood-fill (which explores many cells) compare to random selection, and why might this cost be acceptable for a real-time game?

It has a larger computational cost since it has to actually compute information about its environment. However, this is acceptable because this algorithm has a much higher win rate, since it has an actual strategy.

**Question 6b (Visualization):** After watching the greedy vs random visualization, describe what strategic patterns you noticed in the greedy agent's movement. How does visualizing the space-control heuristic in action deepen your understanding compared to just reading the code?

The greedy agent mostly just goes up and down, filling in lines vertically. Visualizing the heuristic makes it make more sense, that it is just trying to fill up space as tightly as possible.

**Question 7:** Explain why minimax requires an evaluation function at depth limits rather than computing exact game outcomes, and what trade-off this represents between accuracy and computational feasibility. Based on the results, at what depth does minimax start to significantly outperform greedy (if at all)? What does this suggest about the "lookahead horizon" needed in Tron?

Computing exact game outcomes to the very end would grow exponentially and not be computationally feasible. The tradeoff is that the solution won't be as perfect, but it will be a lot more efficient to find. At depth 5, the minimax starts to significantly outperform greedy. This suggests that the lookahead horizon increases the quality of the decisions in Tron.

**Question 8:** Analyze how alpha-beta pruning reduces the search space without affecting the final decision, and describe a scenario where pruning would be most effective (many cutoffs vs few cutoffs).

Alpha-beta pruning prevents searching where there will not be a solution, therefore it still makes the same decision without searching uselessly. It would be most effective with many cutoffs, given the branches that are cut off truly do not affect the final solution.

**Question 9:** Compare minimax's assumption of optimal opponent play to the greedy agent's behavior. When would this assumption hurt minimax's performance, and when would it help?

Minimax assumes the opponent has the same strategy as itself, which it does not. This assumption would hurt minimax's performance when it decides not to take an optimal action because it thinks the opponent will cut it off or otherwise make it a bad action, but the opponent does not do this.

**Question 9b (Critical Thinking):** If minimax with depth-3 only wins 50-60% of games against greedy (rather than 80-90%), what does this suggest about the relationship between lookahead and the quality of the evaluation function? Consider that both algorithms use the same space-difference heuristic at their search horizon.

A higher lookahead value improves the quality of the evaluation function. Both algorithms use the same heuristic, but the one that also looks ahead wins more.

**Question 10:** Explain why MCTS can make good decisions without explicitly evaluating position quality, and how the law of large numbers ensures convergence to optimal play.

MCTS runs a bunch of simulations of what could happen if it made certain decisions, and then it chooses one based on that. The more simulations it runs, the more optimal the play will be.

**Question 11:** Compare the UCB1 exploration parameter's role in MCTS to the depth parameter in minimax. How do they both address the exploration-exploitation trade-off, and what makes their approaches fundamentally different?

UCB1 determines whether to explore new options or use known good moves, whereas minimax depth determines how far to explore before making a decision. Minimax does not run multiple simulations, it just explores and makes a decision once.

**Question 12:** Describe why MCTS might outperform minimax in games with high branching factors or deep game trees, referencing the computational complexity of each approach.

Minimax will need more depth and therefore more memory and compute to solve games with high branching factors or deep game trees. MCTS, on the other hand, just needs to run the game multiple times, and it doesn't really look ahead so the depth doesn't matter as much, and it doesn't explore all options at once so the high branching factor doesn't matter much either.