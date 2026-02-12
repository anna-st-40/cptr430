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