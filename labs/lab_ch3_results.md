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

**Question 5:** Observe the frontier size as the search progresses. Why does the frontier grow so rapidly in BFS, and what does this tell you about the space complexity of this algorithm? How would this affect solving larger problems?

**Question 6:** Compare the number of nodes expanded to the length of the solution path. Why does BFS typically expand many more nodes than are in the final solution? What does this reveal about the efficiency of uninformed search?