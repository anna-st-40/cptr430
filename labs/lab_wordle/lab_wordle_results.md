# Names: Jaime Garcia, Anna Stefaniv Oickle
# Lab: lab3 (Wordle)
# Date: idk

**Question 1:**In the constraint propagation example, explain the relationship between the pattern encoding and the reduction in state space. How does each piece of information (gray, yellow, green) contribute differently to narrowing possibilities?

The pattern encoding consists of a tuple of numbers: a 1 (yellow) means the letter is in the word but not in the correct spot, a 2 (green) means the letter is in the word and in the correct spot, a 0 (gray) means the letter is not in the word. The color gray (0) removes all letters that do not belong to the word (state space), therefore reducing the state space to those words that do not contain those letters.

**Question 2:**The feedback mechanism is deterministic—the same guess against the same answer always produces the same pattern. How does this property affect the agent strategies we'll explore? Would probabilistic feedback change the problem fundamentally?

All the information the agent knows about the pattern encoding will never change. If the same guess against the same answer always produces the same result, the information is not useful because it is the same as the previous attempt. 

**Question 3:**The frequency agent uses a "greedy" approach—it always picks what seems best locally without considering future moves. Describe a scenario where this could lead to suboptimal performance compared to an agent that thinks ahead.

A scenario that could lead to suboptimal performance is if the greedy approach picks a word that none of its letters are in the correct word. An agent that thinks ahead will consider that and not choose a word whose letters are not in the correct word.

**Question 4:**Notice how the unique letter bonus affects word selection. Why might choosing words with five distinct letters be advantageous in early guesses but potentially wasteful in later guesses?

Choosing word with five distinct letter is advantageous in early guesses because by doing so you are covering a wide range of options, checking what letter belong to the right word and what letters do not. You could potentially be wasting attempt in later guesses because by then you should have an idea of what letters belong to the right word, it wouldn't be so right to just guess random 5 letter words.

**Question 5:**This agent recalculates frequencies based on the remaining word list after each guess. How does this adaptive behavior differ from using fixed, pre-computed frequencies? What are the computational trade-offs?

The agent is able to fine-tune its guesses based on discovered constraints. The tradeoff is that we have to recompute every time, which uses computational resources, but at the end it leads to better guesses and potentially getting the right answer with fewer guesses.

**Question 6:**Entropy measures how evenly a guess partitions the remaining possibilities. Explain why a guess that splits possibilities into equally-sized groups has higher entropy than one that creates very uneven partitions. Use a concrete example from the output.

A guess that splits possibilities into equally-sized groups has higher entropy because the search space doesn't really get reduced, you still have two groups of equal size to look in whereas a guess that creates very uneven partitions somewhat reduces entropy and the search space because you know that in one partition the entropy is less than in the other so you have a condensed search space. In the code output, the guesses for the word CIDER, the first guess was SERAI, its entropy was 5.89 but the pattern outcome was somewhat uneven. Then, in the next guess the entropy was reduced to 3,74.

**Question 7:**The entropy agent might choose a word that cannot possibly be the answer if it maximizes information gain. Is this rational? Discuss the trade-off between "playing to win immediately" versus "playing to gather information."

**Question 8:**Compare the computational complexity of the frequency heuristic versus entropy calculation. As the word list grows larger, which approach scales better and why? What does this tell us about the trade-off between optimality and feasibility?