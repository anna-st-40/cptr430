# Names: Anna Stefaniv Oickle, Jaime Garcia
# Lab: Ch3
# Date: Jan 16, 2026

**Question 1:** Why is it important to have a formal problem formulation before applying a search algorithm? What would happen if any of the five components (initial state, actions, transition model, goal test, path cost) were missing or incorrectly defined?

If any of the five components are missing, the algorithm would not have a way to know what to do. If they are incorrectly defined, the algorithm will not have accurate information to make the best descisions (or perhaps even any decisions that are valid).

**Question 2:** In the 8-puzzle, we represent states as tuples of 9 numbers. What are the advantages of this representation compared to using a 2D array or other data structures? Consider both computational efficiency and ease of use.

That way, we only need to use one index to access the elements, which is simpler than using 2 indices for a 2D array.

**Question 3:** The transition model in our implementation checks whether moves are valid before executing them. Why is it important for the `get_actions()` method to only return valid actions rather than returning all possible actions and letting the search algorithm handle invalid ones?

If we only let the algorithm handle valid actions, it increases computational efficiency since it is not wasting time looking at actions it can't take anyways.

**Question 4:** BFS explores nodes level by level (all nodes at depth 1, then all at depth 2, etc.). How does this exploration strategy guarantee that BFS finds the optimal solution for problems where all actions have the same cost?

It explores all paths of the same length at the same time (starting from the shortest), so the first path it finds will also be the shortest.

**Question 5:** Observe the frontier size as the search progresses. Why does the frontier grow so rapidly in BFS, and what does this tell you about the space complexity of this algorithm? How would this affect solving larger problems?

When nodes have multiple children, each exploration takes one node off the frontier but also adds multiple more to the back of the queue, so the frontier just keeps growing. The space complexity is not constant, and grows according to the average number of children per node. It would be really slow in solving larger problems.

**Question 6:** Compare the number of nodes expanded to the length of the solution path. Why does BFS typically expand many more nodes than are in the final solution? What does this reveal about the efficiency of uninformed search?

It is looking at all possible solutions up to a certain path length (the first shortest solution). It finds a very efficient solution, but the process to find that solution is not very efficient, as it has to visit a lot of nodes.

**Question 7:** Compare the maximum frontier size between DFS and BFS. Why does DFS use significantly less memory, and in what situations would this memory advantage be crucial for solving a problem?

Unlike BFS, DFS does not store the entire level in memory at once. This would be advantageous in scenarios where the solution is deep in a wide tree or a graph with a large branching factor.

**Question 8:** DFS found a solution, but was it the optimal (shortest) solution? Explain why DFS is not guaranteed to find the optimal solution even when one exists, and describe a scenario where DFS might find a very poor solution.

No, it was not the optimal solution. It just finds the first leftmost solution, which is much faster than finding the absolute optimal solution. DFS might find a very poor solution when there is a very deep solution somewhere in the left side of the tree, while a very shallow solution exists on the right side of the tree.

**Question 9:** We implemented a depth limit to prevent DFS from exploring infinitely deep paths. What problems could arise without this limit, and how does this relate to the concept of completeness in search algorithms?

Without this limit, it can get stuck with infinite exploration, since the tree is infinite and unless the solution exists along the leftmost branch (constantly taking only that one action), it will search forever. There is no guarantee that the algorithm will find a solution if it exists.

**Question 10:** How does UCS decide which node to expand next, and why does this strategy guarantee finding the optimal solution? Compare this to how BFS makes its expansion decisions.

It expands the node with the lowest cost to travel to it first, and puts it on a priority queue. The solution you find will be the lowest cost solution because you are always looking at the currently lowest path first. This is similar to BFS in that it orders the nodes into the frontier by cost, but in BFS it is implied that the cost is always 1, so the path cost lines up with the path length and the level.

**Question 11:** Observe the path found by UCS through the grid. Does it go directly toward the goal, or does it take a longer route? Explain why UCS chose this path in terms of path cost versus path length.

It does not go directly, it takes a longer route. However, that longer route costs less than a direct route. The "direct" path would cost 11, while the optimal path costs 8.

**Question 12:** In what types of problems would UCS perform identically to BFS? When would UCS clearly outperform BFS in terms of solution quality? Give specific examples of problem domains.

It would perform identically to BFS where all the path costs are the same. An example where UCS outperforms would be a GPS navigator where the speed limit of roads matters, not just the actual length of the road. In general, it would outperform in any problem where the path costs are vastly different.