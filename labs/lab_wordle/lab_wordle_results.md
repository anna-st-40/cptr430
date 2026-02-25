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

**Question 9:** The minimax agent provides a guarantee: "No matter what the answer is, I'll have at most X words remaining after this guess." Why might this guarantee be valuable in competitive or time-limited scenarios even if average performance is worse?

We know the lower limit of how many words we remove from the search space and the upper limit of how many words we will have at maximum. This is valuable because with each guess the agent makes, we are reducing our search space and trying to minimize worst-case scenario.

**Question 10:** Observe cases where entropy and minimax choose different first words. Describe the philosophical difference in these strategies: one is optimistic (optimizing average case) and one is pessimistic (optimizing worst case). Which worldview seems more appropriate for Wordle, and why?

The optimistic view tries to optimize the average case, which means using the words that provide useful information to the solving process. Pessimistic view optimizes the worst case, which discards the words that are not useful to the problem. For Wordle, a pessimistic approach is better because after each guess we minimize our search space; the amount of words that we have available to choose from whereas the optimistic agent chooses words from the same or slightly reduced search space.

**Question 11:** As the number of remaining possibilities decreases (e.g., down to 2-3 words), do you notice the strategies converging? Explain why different optimization criteria matter more when uncertainty is high versus low.

Yes, the strategies converge. When uncertainty is high, we want to reduce the search space and different optimization critieria take different approaches. When uncertainty is low, we want to get as close as posssible to the actual right answer, and both optimization criteria achieve the same result at low uncertainty.

**Question 12:** The hybrid agent switches strategies mid-game. Describe what fundamentally changes about the problem when you have 10 remaining possibilities versus 2 remaining possibilities that justifies this strategy change.

Entropy takes much longer to execute, minimax is generally faster. When we have 10 remaining possibilities, the search space is still considerably large whereas when we have 2 remaining possibilities, we have 50% chance of choosing the right answer. Entropy is better at info gathering and minimax is better at guaranteeing success, this is why minimax is used at the end whhen we have a higher chance of success.

**Question 13:** The threshold parameter (when to switch strategies) is somewhat arbitrary. Propose a method for automatically determining the optimal threshold based on the statistical properties of the word list. What factors should influence this decision?

A better method for automatically determining the optimal threshold could be something like: take the average of the 5 most common letters in the remaining word list and use that as a threshold. What should influence is the count of how many times each letter apprears in each word, rank them and then take the average of that.

**Question 14:** Could we extend this hybrid approach to three or more strategies? Sketch out a meta-strategy that might incorporate frequency heuristics, entropy, and minimax at different stages. What would be the benefits and costs?

Frequency heuristics could be use at the very beginning to kind of narrow down our search space by choosing very common words, then entropy to gather information and minimax when our search space is very narrow and we only have a few remaining possibilities. The benefits should be that the problem should be solved in less steps but it would require more computational expenses as it has to use computer resources for 3 different strategies rather than 2.
