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