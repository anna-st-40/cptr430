# Names: Jaime Garcia, Anna Stefaniv Oickle
# Lab: lab6 (NLP)
# Date: March 9

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

**Question 12:** For the analogy man:king :: woman:?, did the model return "queen" as the top answer? Look at the full top-5 list — are the other answers plausible or surprising? What does this tell you about what the model has actually learned vs. what we might hope it learns?

Yes, it did return queen as the top answers. The ones that are surprising to us are daughter, throne and monarch because they didn't really make sense to us, the other ones did. All the words are related, they just don't make perfect sense in the context, but not so much in the exact correlation between the words.

**Question 13:** In the bias probe, which professions are most strongly associated with "man," and which with "woman"? Do these associations reflect linguistic reality (how these words actually co-occur in text) or social stereotypes — or both? Why is it difficult to separate the two?

The profession that is most strongly associated with man is doctor, however, woman has a larger similarity to doctor than men, which is surprising. The largest difference that favors a man is CEO, about 0.1 above the similarity of a woman. The largest difference between a woman and man profession is nurse. For women it is about 1.6 above men. These associations reflect social stereotypes, but linguistic reality also reflects social stereotypes.

**Question 14:** List the three tokens with the highest saliency scores. Are they the words you would intuitively consider most important for determining sentiment? If there are surprises (e.g., punctuation or function words scoring high), propose a hypothesis for why the model might be sensitive to them.

Disaster is the highest, followed by brilliant, and the last one is the CLS token. Disaster and brilliant are two important words that demonstrate sentiment. We saw in previous exercises that the CLS token stores a bunch of meaning in them, every token attends the CLS token.

**Question 15:** This sentence contains both positive ("brilliant") and negative ("disaster") sentiment words. Which scores higher in saliency? What does this tell you about how the model balances conflicting sentiment signals, and does it match the model's final prediction?

Disaster scores higher than brilliant. The model learns that the last thing that was said or mentioned in the sentence is more importantly and therefore has more weight. In the sentence, it is mentioned that the film was brilliant. This sits in the beginninng of the sentence. The conclussion of the sentence is that the ending was a disaster. We as humans always remember the last part of the sentence, and the model learns from that.

**Question 16:** Which masked word caused the largest drop in confidence? Did any masking cause the predicted label to flip? What do these results tell you about the relative causal importance of different tokens compared to what saliency scores alone predicted?

The word "disaster" caused the largest drop in confidence, which also caused the predicted label to flip. Both brilliant and disaster have high saliency scores, however, masking brilliant changed almost nothing. This sugggests that saliency scores alone don't accurately indicate the cause of importance.

**Question 17:** Masking "was" likely had little effect on the prediction, yet some function words may have had non-trivial saliency scores in Exercise 6. How do you reconcile these two findings? What does this suggest about the difference between gradient sensitivity and causal importance?

Just because has gradient sensitiity doesn't mean it's actually causing the meaning to change.

**Question 18:** For the pairs where adding or removing "not" changes the label, is the confidence shift large or small compared to swapping a positive word for a negative one (e.g., "excellent" → "terrible")? What does the relative magnitude of these confidence changes reveal about how the model weights negation versus lexical sentiment signal?

The confidence shift was small for al of them. There's a specific case in sentence 5 where the edit was that the word "didn't" was removed, but the label stayed the same, however the confidence level shifted by a considerable amount. Changing from positive to negative or viceversa words didn't really change the confidence level but the label changes. Coming back to sentence 5, the reason why the label didn't change is because the sentence contained another adjectives that were negative that the model decided had more weight, so although the edit removed the "not", the label stayed negative due to that.

**Question 19:** No. It does not change at all. Confidence level doesn't change, label doesn't change. It is doing a more hollistic representation because removing the adjective didn't impact the result significantly.