# Names: Jaime Garcia, Anna Stefaniv Oickle
# Lab: Lab Chapter 6
# Date: Feb 12, 2026

**Question 1:** Sudoku has 81 variables (cells) but only 27 constraints (9 rows + 9 columns + 9 boxes). Explain why each constraint actually represents multiple binary constraints between pairs of cells. How many total binary constraints exist in a 9×9 Sudoku?

We have one constraint per row, column and box. However, each constraint has 36 binary constraints representing every possible combination of two numbers, calculates by the combinations formula. So, the total number of binary constraints that exist in a 9x9 Sudoku grid is 27x36 = 972.

**Question 2:** A Sudoku puzzle is only solvable if it has exactly one solution. From a CSP perspective, what would it mean for a puzzle to have zero solutions versus multiple solutions? How would you detect these conditions?

If a puzzle has zero solutions, then this means that the contraints conflic and there is no way to satisfy all the constraints. If a puzzle has multiple solutions, then this means that there is more than one way to satisfy all the constraints.

**Question 3:** Naive backtracking explores the search tree depth-first without any lookahead. Describe a scenario where this leads to significant wasted work—specifically, where the algorithm makes many assignments before discovering a conflict that could have been detected earlier.

A scenario would be when the initial grid is sparse. The algorithm will start exploring the tree, trying different solutions in each cell and seeing which one is consistent, but since our initial grid is sparse it may have put a wrong number early on and not discover the mistake until much later because our grid didn't really give us the correct information.

**Question 4:** The solver selects variables in a fixed order (left-to-right, top-to-bottom). Explain why this ordering is arbitrary and potentially inefficient. What information about the CSP state would be useful for choosing which variable to assign next?

Choosing a column, row, or box with the least amount of empty cells would be more efficient than just selecting variables in a fixed order like the current solver does. In a column, row, or box with very few amount of empty cells, the branching factor for each cell is reduced whereas in an arbitrary selection, the branching factor is uncertain.

**Question 5:** Notice that the number of assignments grows dramatically for harder puzzles. Using the branching factor concept, explain why CSP search complexity is exponential. If each empty cell has an average of 5 possible values, approximately how many nodes would be explored in the worst case for a puzzle with 50 empty cells?

The more empty cells we have, the larger the branching factor is. An increased branching factor means we will have expontentially more children.
If we have 50 empty cells and for each empty cell we have 5 options, the worst case would be that we have to fully explore every single branch for every single cell, which would be approximately $ 5^{50} $

**Question 6:** Forward checking maintains arc consistency between the current variable and its neighbors. Explain what happens when a neighbor's domain becomes empty (domain wipeout). Why is this more efficient than discovering the conflict through multiple future assignments?

When a neighbor's domain becoes empty, there's two options: either you have satisfied that constraint for that row, column, or box, or your puzzle is unsolvable in the current state and you need to bracktrack, find where you made the mistake and fix it. This is. more efficient because the conflict is discovered early on in the traversing process and you don't actually have to move to that cell, you just check the domain of your neighbor cells for domain wipeout.

**Question 7:** Notice the "domain reductions" metric. Each reduction eliminates a potential branch of the search tree. Calculate approximately how many search nodes were pruned by forward checking in the example. How does this relate to the speedup observed?

1480 search nodes were approximately pruned by forward checking in the example. Since a very large number of nodes was removed, this means we have to traverse and search less nodes, which leads to an increased search and completion speed.

**Question 8:** The AC-3 algorithm uses a queue to process arcs. Explain why arcs must be re-added to the queue when domains change. What would happen if we only processed each arc once without re-queuing?

Arc consistency means that for every pair of constrained variables (X, Y), each value in X's domain must have a compatible value in Y's domain. Arcs must be re-added to the queue when domains change because the change in domain can break the previously established consistency of the neighbor variables. If we didn't do this, the algorithm would fail to check that the final result is consistent across all constraints.

**Question 9:** Some easy Sudoku puzzles can be solved by AC-3 alone without any search. Explain what this reveals about the puzzle's structure. What property must a puzzle have for AC-3 to solve it completely through constraint propagation?

If a puzzle can be completed by AC-3 alone, this means that the puzzle's structure has a high local consistency, it doesn't have hidden patterns that an algorithm is unable to recognize. A property that a puzzle must have for AC-3 to solve it completely through constraint propagation is that there must be a unique solution to the puzzle, that way there's no confusion in the algorithm.

**Question 10**: The MRV heuristic chooses the "most constrained" variable first, which seems counterintuitive—why tackle the hardest decisions first? Explain the "fail-first" principle and how detecting failures early reduces total search effort.

The fail-first principle states that it can detect failures as early as possible by tackling hard decisions first instead of waiting until the end. Imagine that we start with the easy problems first, and say we are able to do all those. Then, once we get to the hard ones, if we make a mistake, then we need to go back and correct all the ones we already "solved", which in turn takes more work.

**Question 11:**
The LCV heuristic orders values by how little they constrain neighbors. Describe the philosophical difference between MRV (fail-first for variables) and LCV (succeed-first for values). Why do these opposite strategies work well together?

MRV takes the hard variables first, tries to solve them. and if there's a problem, then it detects it early on. LCV, after choosing a value, order the values by how many options they leave for neighbor cells, so it wants to keep its options open. These appear to be opposite strategies, but they work well together because they achieve complimentary goals. MRV finds a problem fast and LCV finds a solution fast. In general, they both work towards findind whether there's a problem or a solution "fast".

**Question 12:** The degree heuristic breaks ties in MRV by choosing variables with the most unassigned neighbors. Explain the reasoning: why would constraining more variables make a variable selection better? Consider both immediate effects and future search reduction.

Constraining more variables leaves less options for another variables, which turns out to be better since you have less options to choose from. If you constraint a lot of variables, then there will be some variables that will only have 1 or 2 values to choose from. This is easier to solve than having all 9 possible values. The inmediate effect is that is an easy solution, in future search reduction this might lead to problems.

**Question 13:** Observe how the performance gap between naive backtracking and intelligent techniques grows with puzzle difficulty. Explain why the benefit of heuristics and constraint propagation is more pronounced on harder puzzles. What does this reveal about the structure of the search space?

Naive backtracking assigns and checks if a certain value works, intelligent techniques explore their domain and their neighbor's domains before assigning or making a conclussion. Therefore, on harder puzzles there's more benefir in using heuristics and constraint propagation techniques. The search space should be the same for all algorithms, but the algorithms take different approaches to solve the puzzle.

**Question 14:** The "hard" puzzle has only 21 givens compared to 36 for the easy puzzle. Discuss why fewer givens doesn't always mean harder—the pattern and distribution of givens matters. What makes a Sudoku puzzle truly "hard" from a CSP perspective?

It't not so much about the givens but more about the constraints, and this is what makes a Sudoku puzzle truly hard from a CSP perspective. Fewer givens doesn't always means harder because as we mentioned, it's more about the contraints we have. If we were given all the 1s, 2s or 3s, then that's gonna be more difficult to solve than if we were given more numbers spread across the board.

**Question 15:** We would have a combination of techniques. For easy and mediums problems we would choose the AC-3 algorithm because it performs really well. Then for hard problems, heuristics performs the best.

**Question 16:** The min-conflicts heuristic chooses values that minimize immediate conflicts. Compare this greedy, local decision-making to backtracking's systematic exploration. Why does backtracking's ability to undo multiple decisions give it an advantage for CSPs?

Because if I made a mistake, it might have been several branches up, so I need to undo several assignments, which is essential for CSPs

**Question 17:** Local search performs well for some CSPs (like N-queens) but poorly for Sudoku. Hypothesize about what properties make a CSP amenable to local search versus requiring systematic search. Consider factors like constraint density, solution density, and landscape topology.

N-queens has multiple solutions and few constraints, whereas Soduku has fewer solutions adn many constraits that interact with each other.

**Question 18:** Z3 uses a declarative approach where you specify constraints and let the solver find solutions. Compare this to the procedural backtracking approaches from earlier exercises. What are the advantages and disadvantages of each paradigm for solving CSPs?

Z3 gets a layer of abstraction above the algorithm approaches we studied earlier. We don't know or implement the internals of Z3, so we just trust that it'll perform well whereas with the other algorithms we have to implement them by hand and we can tweak them and make improvements if they perform poorly.

**Question 19:** The Distinct() constraint in Z3 is a built-in primitive that handles "all different" efficiently. Earlier exercises checked constraints pairwise manually. Explain how a solver could optimize the Distinct() constraint internally using techniques beyond pairwise checking. Consider arc consistency and conflict analysis.

If you check pairwise that's gonna be O(n^2) complexity, so we could choose a better algorithm that its time complexity is less than checking pairwise. There's an algorithms called bipartite checking that performs better than pairwise checking, and its time complexity is less than O(n^2)

**Question 20:** Z3 uses techniques like conflict-driven clause learning (CDCL) which learns from failures to avoid repeating similar mistakes. Compare this to plain backtracking which forgets why branches failed. How does learning from conflicts relate to the constraint propagation and heuristics you've explored in previous exercises?

Learning from conflicts avoid making future mistakes. If you made a mistake early, then you learn from it and if you encounter something similar again it later on then you know what to do. Plain backtracking just picks a random value to try, if it fails, it corrects it, but it doesn't keep track of its mistakes.