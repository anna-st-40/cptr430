# Names: Jaime Garcia, Anna Stefaniv Oickle
# Lab: lab6 (NLP)
# Date: Today's date

**Question 1:** Looking at the attention matrix for layer 1, head 1, which token receives the most attention from the word "it"? Does this match your linguistic intuition about what "it" should refer to in the sentence? Why or why not?

In the attetion matrix for layer 1, head 1. The visualization doesn't really provide much information as to token attention since we're so early in the layer-head relationship. It's not very intuitive to determine whether what token is attending what token

**Question 2:** The [CLS] token often accumulates disproportionately high attention. What role does [CLS] play in BERT's architecture, and why might many tokens attend strongly to it even though it carries no dictionary meaning?

The [CLS] token is prepended to every input sequence that acts as a global representation of the entire sequence. Tokens attend strongly to it because is a central register where information from all other tokens is added.

**Question 3:** Attention is sometimes described as a "soft lookup" — rather than retrieving one fixed answer, the model blends information from many tokens weighted by relevance. How does this differ from how a traditional rule-based parser would resolve pronoun reference?

Attention mechanism replaces the rigid grammatical rules of traditional rule-based parser with a weighted, continuous, and probabilistic approach to pronoun resolution. Attention blends information from all potential candidates.

**Question 4:** After exploring multiple heads in layer 1, describe two heads that appear to capture qualitatively different relationships. What linguistic patterns do you think each is tracking, and what evidence in the visualization supports your interpretation?

The first and red head appear to capture different relationships. The first head just tried to capture and relate each token, and in the red layer (4th one from left to right), it's relating tokens to the preceeding token. Since we're so early in the process, layer 1 is trying to find relationships between tokens, trying to get a feel of what there is available.

**Question 5:** Compare the overall "texture" of attention in layer 1 vs. layer 8 vs. layer 12. In which layer do attention patterns look more diffuse (spread across many tokens) vs. more focused? What might this difference tell us about the progression from syntactic to semantic processing?

The layer that the attention pattern is more diffuse or spread across many tokens is layer 1. The layer that is more focused is layer 8, everything relates to [SEP] and to some words that are gramatically related. Layer 12 is just punctuation, everything relates to the punctuation marks. Early in the process you are just exploring the tokens available, relating them to each other and trying to find relations. As we progress, we give the relationships meaning.

**Question 6:** The AIMA book describes intelligence as the ability to act appropriately given incomplete information. In what sense does multi-head attention allow a model to handle ambiguity — like the ambiguous "it" in our sentence — better than a single attention head would?

It looks at it from different angles, what different thing might mean and realte to.

**Question 7:** In layer 12, does "it" consistently attend more to "trophy" than "suitcase" across all heads? Are there heads where "suitcase" dominates? What does this variability across heads suggest about how BERT distributes reasoning across its components?

For the most part, "it" attends to trophy more. On the layers that "it" attends to trophy and suitcase a lot, it attends to trophy more than it does to suitcase. Suitcase dominates in heads 3, 6, and slightly 8. It reasons differently across the different heads to get a full picture.

**Question 8:** Attention weight alone does not prove a model has "understood" coreference — it is a correlation, not a causal explanation. What additional experiment could you design to test whether BERT truly resolves this Winograd Schema correctly?

We mask "trophy" and "suitcase" and check if the attention weight changes significantly.

**Question 9:** The Winograd Schema was proposed as a test of machine intelligence that is trivial for humans but hard for simple statistical models. Based on what you've observed in these attention exercises, do you think BERT "understands" this sentence in a meaningful sense? Defend your position using specific observations from the lab.

Yes. It is able to figure out what "it" is reffering to meaningfully. In the context of the sentence and in spoken language, we humans know that the sentence refers to the trophy, and this model has been able to successfully guess that "it" refers to either suitcase or trophy, and the correct word is in the list.

**Question 10:** Describe the spatial organization of the three word clusters in your plot. Are they clearly separated, or do any categories overlap? Which two categories are most geometrically similar to each other, and why might that be?

Yes, they are clearly separated. The two categories that are most geometrically similar to each other are animals and professions because these two are more likely to be used as subjects in a sentence than the countries cluster.

**Question 11:** The distributional hypothesis underpins word embeddings: words are similar if they appear in similar contexts. Can you think of a pair of words that would be incorrectly placed near each other by this principle (false semantic neighbor), or a pair incorrectly placed far apart (false semantic distance)? What does this suggest about the limits of corpus-based meaning?

Probably homographs because they are word that have multiple meanings depending on the context. For example, bat (animal/sport), "I wil park the car in the park".