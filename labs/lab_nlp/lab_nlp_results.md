# Names: Jaime Garcia, Anna Stefaniv Oickle
# Lab: lab6 (NLP)
# Date: Today's date

**Question 1:** Looking at the attention matrix for layer 1, head 1, which token receives the most attention from the word "it"? Does this match your linguistic intuition about what "it" should refer to in the sentence? Why or why not?

In the attetion matrix for layer 1, head 1. The visualization doesn't really provide much information as to token attention since we're so early in the layer-head relationship. It's not very intuitive to determine whether what token is attending what token

**Question 2:** The [CLS] token often accumulates disproportionately high attention. What role does [CLS] play in BERT's architecture, and why might many tokens attend strongly to it even though it carries no dictionary meaning?

The [CLS] token is prepended to every input sequence that acts as a global representation of the entire sequence. Tokens attend strongly to it because is a central register where information from all other tokens is added.

**Question 3:** Attention is sometimes described as a "soft lookup" — rather than retrieving one fixed answer, the model blends information from many tokens weighted by relevance. How does this differ from how a traditional rule-based parser would resolve pronoun reference?

Attention mechanism replaces the rigid grammatical rules of traditional rule-based parser with a weighted, continuous, and probabilistic approach to pronoun resolution. Attention blends information from all potential candidates.