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