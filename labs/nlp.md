# Natural Language Processing: How Models Understand Language Lab
**Reference: Russell & Norvig, *Artificial Intelligence: A Modern Approach*, Chapter 23 (Natural Language Processing)**

---

## Learning Objectives

By the end of this lab, you will be able to:

1. **Identify** how transformer attention mechanisms distribute focus across tokens in a sentence
2. **Distinguish** how attention patterns evolve across different layers of a neural network
3. **Explain** how word embeddings encode semantic relationships as geometric structure in vector space
4. **Classify** whether observed embedding clusters reflect linguistic categories or learned biases
5. **Analyze** which tokens most influence a model's sentiment classification decision
6. **Compare** model behavior before and after removing high-salience tokens to evaluate prediction sensitivity

---

## Lab Overview

This lab takes an **observation-first** approach: you will run fully implemented code and study what it reveals about how modern NLP models work internally. Rather than building models from scratch, you'll develop intuitions by watching them in action — visualizing attention, exploring geometric meaning, and probing what drives classification decisions.

**Python 3.10+ is recommended.** Install dependencies before starting:

The lab is organized into three thematic modules:
- **Module A** — Attention Visualization (Exercises 1–3)
- **Module B** — Word Embedding Geometry (Exercises 4–5)
- **Module C** — Gradient Saliency & Model Sensitivity (Exercises 6–7)

---

I suggest running the following commands from your base user directory:

```bash
mkdir cs430 
cd cs430 
uv add transformers torch bertviz gensim scikit-learn matplotlib umap-learn captum
source .venv/bin/activate
touch nlp.ipynb
```

The last command will create a file such as `nlp.ipynb`. 

---

#### uv

I highly recommend uv (https://docs.astral.sh/uv/). It (according to their docs):

- 🚀 Is a single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
- ⚡️ Is 10-100x faster than pip.
- 🗂️ Provides comprehensive project management, with a universal lockfile.
- ❇️ Runs scripts, with support for inline dependency metadata.
- 🐍 Installs and manages Python versions.

---

#### Jupyter Notebook

This lab is designed to be run in a Jupyter notebook environment, because the examples build progressively.

Select the virtual environment created by `uv`` (`cs430`) as the kernel for your Jupyter notebook.

Paste the code for each exercise in a new code cell.

If you can't use Jupyter Notebook for whatever reason, just build up a regular Python program, and ignore output from earlier exercises.

#### Submission 

Make sure to record your answers to *all* reflections to submit at the end of the lab!

---


## Exercise 1: Your First Look at Attention

### Description
Transformer models like BERT process language using *attention* — a mechanism that lets each token look at every other token and decide how much to "focus" on it. This exercise introduces you to raw attention weights so you can see this process before adding any visualization layers.

### Key Concepts
- **Token**: The basic unit BERT processes — roughly a word or subword (e.g., "playing" might split into "play" + "##ing")
- **Attention weight**: A number between 0 and 1 representing how much one token attends to another; weights for a given token sum to 1
- **Attention head**: BERT runs multiple attention computations in parallel; each "head" can learn different relationships
- **Layer**: BERT has 12 stacked transformer layers; early layers tend to capture syntax, later layers semantics
- **`[CLS]` / `[SEP]`**: Special tokens BERT adds to mark sentence start and end

### Task
Run the code and observe the printed attention matrix. Focus on:
- Which token-to-token pairs have the **highest weights** in layer 0, head 0
- How the `[CLS]` and `[SEP]` tokens behave compared to content words
- Any asymmetries (token A attending to B ≠ token B attending to A)

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

sentence = "The trophy doesn't fit in the suitcase because it is too big."
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)

# Shape: (layers=12, heads=12, seq_len, seq_len)
attentions = outputs.attentions

# Inspect layer 0, head 0
layer, head = 0, 0
attn_matrix = attentions[layer][0][head].numpy()

print("Tokens:", tokens)
print(f"\nAttention matrix — Layer {layer+1}, Head {head+1}")
print(f"{'':>12}", end="")
for t in tokens:
    print(f"{t:>10}", end="")
print()

for i, src in enumerate(tokens):
    print(f"{src:>12}", end="")
    for j in range(len(tokens)):
        print(f"{attn_matrix[i][j]:>10.3f}", end="")
    print()
```

### Reflection Questions

**Q1.** Looking at the attention matrix for layer 1, head 1, which token receives the most attention from the word "it"? Does this match your linguistic intuition about what "it" should refer to in the sentence? Why or why not?

**Q2.** The `[CLS]` token often accumulates disproportionately high attention. What role does `[CLS]` play in BERT's architecture, and why might many tokens attend strongly to it even though it carries no dictionary meaning?

**Q3.** Attention is sometimes described as a "soft lookup" — rather than retrieving one fixed answer, the model blends information from many tokens weighted by relevance. How does this differ from how a traditional rule-based parser would resolve pronoun reference?

---

## Exercise 2: Visualizing Attention Across All Heads

### Description
A single attention head sees only one "view" of the sentence. BERT runs 12 heads per layer simultaneously, each potentially learning different linguistic relationships — syntactic agreement, coreference, semantic similarity, and more. Here you'll use `bertviz` to see all heads at once.

### Key Concepts
- **Multi-head attention**: Running several attention functions in parallel, each with its own learned weight matrices
- **Head specialization**: Different heads empirically learn different patterns (e.g., one head may track subject-verb agreement while another tracks dependency arcs)
- **Attention rollout**: A technique to trace how attention flows through all layers cumulatively, not just one layer at a time
- **Interactive visualization**: `bertviz` produces an HTML widget; darker lines = stronger attention

### Task
Run the code to produce an interactive `bertviz` visualization. In the widget:
- Select **Layer 1** and look at all 12 heads — notice how different they are from each other
- Then switch to **Layer 12** and compare
- Try clicking individual tokens to highlight their outgoing attention

```python
from transformers import BertTokenizer, BertModel
from bertviz import head_view
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

sentence = "The trophy doesn't fit in the suitcase because it is too big."
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)

# Opens an interactive visualization in Jupyter
# In a plain script, bertviz will emit an HTML string
head_view(outputs.attentions, tokens)
```

> **Note:** Run this in a Jupyter notebook for the interactive widget. If running as a script, `bertviz` will emit HTML you can save and open in a browser.

### Reflection Questions

**Q4.** After exploring multiple heads in layer 1, describe two heads that appear to capture qualitatively different relationships. What linguistic patterns do you think each is tracking, and what evidence in the visualization supports your interpretation?

**Q5.** Compare the overall "texture" of attention in layer 1 vs. layer 12. In which layer do attention patterns look more diffuse (spread across many tokens) vs. more focused? What might this difference tell us about the progression from syntactic to semantic processing?

**Q6.** AIMA describes intelligence as the ability to act appropriately given incomplete information. In what sense does multi-head attention allow a model to handle ambiguity — like the ambiguous "it" in our sentence — better than a single attention head would?

---

## Exercise 3: Tracing Coreference — Does "it" Find "trophy"?

### Description
The sentence *"The trophy doesn't fit in the suitcase because it is too big"* is a classic Winograd Schema — a test of commonsense reasoning. The correct antecedent of "it" is "trophy." Here you will measure directly whether BERT's attention reflects this resolution.

### Key Concepts
- **Coreference resolution**: Determining which noun phrases in a text refer to the same real-world entity
- **Winograd Schema**: Pronoun resolution problems that require world knowledge, not just local syntax
- **Antecedent**: The noun that a pronoun refers back to
- **Attention as proxy for reasoning**: High attention from "it" → "trophy" would suggest the model has internalized this coreference relationship

### Task
Run the code and study the bar chart. Observe:
- Which token "it" attends to most strongly across all heads in layer 12
- Whether "trophy" consistently outranks "suitcase"
- Which heads show the clearest coreference signal

```python
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import numpy as np

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

sentence = "The trophy doesn't fit in the suitcase because it is too big."
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)

it_idx       = tokens.index("it")
trophy_idx   = tokens.index("trophy")
suitcase_idx = tokens.index("suitcase")

layer = 11  # Layer 12 (0-indexed)
attn  = outputs.attentions[layer][0]  # shape: (heads, seq, seq)

trophy_attn   = attn[:, it_idx, trophy_idx].numpy()
suitcase_attn = attn[:, it_idx, suitcase_idx].numpy()

heads = np.arange(1, 13)
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(heads - width/2, trophy_attn,   width, label='"trophy"',   color="steelblue")
ax.bar(heads + width/2, suitcase_attn, width, label='"suitcase"', color="coral")
ax.set_xlabel("Attention Head")
ax.set_ylabel('Attention from "it"')
ax.set_title('Layer 12: How much does "it" attend to each candidate antecedent?')
ax.set_xticks(heads)
ax.legend()
plt.tight_layout()
plt.savefig("coreference_attention.png", dpi=150)
plt.show()
print("Saved: coreference_attention.png")
```

### Reflection Questions

**Q7.** In layer 12, does "it" consistently attend more to "trophy" than "suitcase" across all heads? Are there heads where "suitcase" dominates? What does this variability across heads suggest about how BERT distributes reasoning across its components?

**Q8.** Attention weight alone does not prove a model has "understood" coreference — it is a correlation, not a causal explanation. What additional experiment could you design to test whether BERT truly resolves this Winograd Schema correctly?

**Q9.** The Winograd Schema was proposed as a test of machine intelligence that is trivial for humans but hard for simple statistical models. Based on what you've observed in these attention exercises, do you think BERT "understands" this sentence in a meaningful sense? Defend your position using specific observations from the lab.

---

## Exercise 4: The Geometry of Meaning — Word Embeddings in 2D

### Description
Before transformers, NLP relied on *static word embeddings* like Word2Vec and GloVe — dense vectors learned so that semantically similar words cluster together in vector space. This exercise projects those high-dimensional vectors into 2D so you can literally see the geometry of meaning.

### Key Concepts
- **Word embedding**: A dense vector (typically 50–300 dimensions) representing a word's meaning, learned from co-occurrence statistics
- **Distributional hypothesis**: Words that appear in similar contexts tend to have similar meanings — the foundation of embedding learning
- **PCA (Principal Component Analysis)**: A linear dimensionality reduction technique that preserves global structure
- **t-SNE / UMAP**: Nonlinear techniques that preserve local neighborhoods; excellent for revealing clusters
- **Vector arithmetic**: Famous property where `king − man + woman ≈ queen` holds in well-trained embedding spaces

### Task
Run the code and examine the scatter plot. Look for:
- Whether the **animal**, **profession**, and **country** clusters are visually separated
- Words that appear in unexpected positions
- The overall shape and spread of the embedding space

```python
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("Loading GloVe embeddings (this may take a minute)...")
glove = api.load("glove-wiki-gigaword-100")

words = {
    "animals":     ["dog", "cat", "horse", "elephant", "lion", "tiger", "wolf", "bear"],
    "professions": ["doctor", "lawyer", "teacher", "engineer", "nurse", "pilot", "chef", "scientist"],
    "countries":   ["france", "germany", "japan", "brazil", "egypt", "canada", "india", "mexico"],
}

all_words, all_labels, all_vecs = [], [], []
colors = {"animals": "steelblue", "professions": "coral", "countries": "mediumseagreen"}

for group, word_list in words.items():
    for w in word_list:
        if w in glove:
            all_words.append(w)
            all_labels.append(group)
            all_vecs.append(glove[w])

vectors = np.array(all_vecs)
pca     = PCA(n_components=2)
coords  = pca.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(10, 7))
for i, (word, label) in enumerate(zip(all_words, all_labels)):
    ax.scatter(*coords[i], color=colors[label], s=80, zorder=3)
    ax.annotate(word, coords[i], fontsize=9, xytext=(4, 4), textcoords="offset points")

for group, color in colors.items():
    ax.scatter([], [], color=color, label=group.capitalize(), s=80)
ax.legend(fontsize=11)
ax.set_title("GloVe Word Embeddings — PCA Projection to 2D", fontsize=13)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.tight_layout()
plt.savefig("embeddings_pca.png", dpi=150)
plt.show()
print("Saved: embeddings_pca.png")
```

### Reflection Questions

**Q10.** Describe the spatial organization of the three word clusters in your plot. Are they clearly separated, or do any categories overlap? Which two categories are most geometrically similar to each other, and why might that be?

**Q11.** PCA can only retain a small fraction of the original 100-dimensional variance in 2D. Looking at the axis labels showing explained variance percentages, what proportion of information is lost in this projection? How should this limitation affect your confidence in conclusions drawn from the 2D plot?

**Q12.** The distributional hypothesis underpins word embeddings: words are similar if they appear in similar contexts. Can you think of a pair of words that would be **incorrectly** placed near each other by this principle (false semantic neighbor), or a pair **incorrectly** placed far apart (false semantic distance)? What does this suggest about the limits of corpus-based meaning?

---

## Exercise 5: Vector Arithmetic and Latent Bias

### Description
One of the most striking properties of word embeddings is that semantic relationships are encoded as consistent vector offsets — enabling "arithmetic of meaning." But the same structure that captures `king − man + woman ≈ queen` also captures social biases present in training corpora.

### Key Concepts
- **Analogy completion**: Finding word `d` such that `a:b :: c:d`, solved as `d ≈ b − a + c` in embedding space
- **Cosine similarity**: Measures the angle between two vectors; values near 1 = similar direction, near 0 = orthogonal
- **Embedding bias**: Systematic associations learned from human-written text that reflect historical stereotypes
- **Debiasing**: Techniques to remove or reduce bias from embeddings while preserving semantic utility

### Task
Run the code and examine both the analogy results and the bias probe. Observe:
- How accurate the analogy completions are (correct word in top 5?)
- The cosine similarities for gendered profession associations
- Whether any associations surprise you

```python
import gensim.downloader as api
import numpy as np

glove = api.load("glove-wiki-gigaword-100")

# ── Analogy Arithmetic ───────────────────────────────────────────────
def analogy(a, b, c, topn=5):
    """Find d such that a:b :: c:d"""
    try:
        return glove.most_similar(positive=[b, c], negative=[a], topn=topn)
    except KeyError as e:
        return f"Word not found: {e}"

print("=" * 50)
print("ANALOGY COMPLETION  (a : b :: c : ?)")
print("=" * 50)
probes = [
    ("man",   "king",   "woman"),
    ("paris", "france", "berlin"),
    ("walk",  "walked", "run"),
]
for a, b, c in probes:
    results = analogy(a, b, c)
    top = results[0][0] if isinstance(results, list) else results
    print(f"  {a}:{b} :: {c}:? → top answer: '{top}'")
    if isinstance(results, list):
        print(f"    Full top-5: {[r[0] for r in results]}")

# ── Bias Probe ───────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("GENDER–PROFESSION BIAS PROBE")
print("Cosine similarity to 'man' vs 'woman'")
print("=" * 50)

professions = ["doctor", "nurse", "engineer", "teacher", "ceo", "secretary", "pilot", "librarian"]

def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

man_vec   = glove["man"]
woman_vec = glove["woman"]

print(f"  {'Profession':>12}  {'sim(man)':>10}  {'sim(woman)':>12}  {'Closer to':>10}")
for prof in professions:
    if prof in glove:
        sm = cosine(glove[prof], man_vec)
        sw = cosine(glove[prof], woman_vec)
        print(f"  {prof:>12}  {sm:>10.4f}  {sw:>12.4f}  {'man' if sm > sw else 'woman':>10}")
```

### Reflection Questions

**Q13.** For the analogy `man:king :: woman:?`, did the model return "queen" as the top answer? Look at the full top-5 list — are the other answers plausible or surprising? What does this tell you about what the model has actually learned vs. what we might hope it learns?

**Q14.** In the bias probe, which professions are most strongly associated with "man," and which with "woman"? Do these associations reflect linguistic reality (how these words actually co-occur in text) or social stereotypes — or both? Why is it difficult to separate the two?

**Q15.** One proposed debiasing approach projects word vectors onto a "gender direction" and subtracts that component. What benefits and potential costs might you anticipate? Consider downstream applications (search, autocomplete) and whether "erasing" bias is the same as "fixing" it.

---

## Exercise 6: Gradient Saliency — Which Words Drive the Prediction?

### Description
Gradient-based saliency maps answer: *if I slightly changed this token's embedding, how much would the model's output change?* Tokens with large gradient magnitudes are the ones the model is most sensitive to — giving us a window into what drives its final decision.

### Key Concepts
- **Saliency map**: A per-token importance score derived from gradients of the output with respect to the input
- **Input × gradient**: Multiplies the gradient by the embedding value for more stable attribution
- **L2 norm**: The length of a vector; used to collapse the per-dimension gradient into a single scalar per token
- **Sentiment classification**: Predicting whether text expresses positive or negative sentiment
- **Interpretability**: The degree to which a model's internal reasoning can be understood by humans

### Task
Run the code and examine the printed saliency scores. Observe:
- Which tokens have the **highest importance** scores
- Whether stopwords ("the", "a", "is") score high or low
- Whether the highest-scoring words intuitively seem sentiment-relevant

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

sentence = "The film was surprisingly brilliant but the ending was a disaster."
inputs   = tokenizer(sentence, return_tensors="pt")
tokens   = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Enable gradient tracking on embeddings
embeddings = model.distilbert.embeddings(inputs["input_ids"])
embeddings.retain_grad()

outputs   = model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"])
pos_score = outputs.logits[0, 1]  # POSITIVE class logit
pos_score.backward()

# Saliency: L2 norm of (gradient × embedding), then normalize
grad     = embeddings.grad[0]
saliency = (grad * embeddings[0].detach()).norm(dim=-1).detach().numpy()
saliency = saliency / saliency.max()

print(f"\nSentence : {sentence}")
print(f"Logits   : {outputs.logits.softmax(-1)[0].tolist()}\n")
print(f"{'Token':>15}  {'Saliency':>8}")
print("-" * 30)
for tok, sal in zip(tokens, saliency):
    bar = "█" * int(sal * 20)
    print(f"{tok:>15}  {sal:>8.4f}  {bar}")
```

### Reflection Questions

**Q16.** List the three tokens with the highest saliency scores. Are they the words you would intuitively consider most important for determining sentiment? If there are surprises (e.g., punctuation or function words scoring high), propose a hypothesis for why the model might be sensitive to them.

**Q17.** This sentence contains both positive ("brilliant") and negative ("disaster") sentiment words. Which scores higher in saliency? What does this tell you about how the model balances conflicting sentiment signals, and does it match the model's final prediction?

**Q18.** Gradient saliency tells us where the model is *sensitive*, not necessarily what it has *learned*. Describe a scenario where a token could have high saliency for the wrong reason — a spurious correlation from training data rather than genuine semantic understanding.

---

## Exercise 7: Perturb and Observe — Removing Key Tokens

### Description
The ultimate test of a saliency score is perturbation: if we remove or mask the "most important" token, does the prediction change dramatically? This exercise connects interpretability theory to empirical testing.

### Key Concepts
- **Ablation**: Systematically removing components of a model or input to measure their causal contribution
- **Prediction confidence**: The probability the model assigns to its chosen class
- **MASK token**: A special token used during BERT pre-training; natural choice for "erasing" a word
- **Causal vs. correlational explanation**: Saliency is correlational; perturbation tests whether the relationship is causal

### Task
Run the code and compare the original prediction to perturbed predictions. Observe:
- How much confidence drops when the top-saliency token is masked
- Whether masking a low-saliency token changes the prediction at all
- Any cases where masking actually *flips* the predicted label

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

sentence = "The film was surprisingly brilliant but the ending was a disaster."

def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(-1)[0]
    label = "POSITIVE" if probs[1] > probs[0] else "NEGATIVE"
    return label, max(probs).item()

def mask_word(text, word):
    return text.replace(word, "[MASK]", 1)

base_label, base_conf = predict(sentence)
print(f"Original  : '{sentence}'")
print(f"Prediction: {base_label} ({base_conf:.4f})\n")

targets = [
    ("brilliant",   "high saliency"),
    ("disaster",    "high saliency"),
    ("was",         "low saliency (expected)"),
    ("surprisingly","medium saliency"),
]

print(f"{'Masked':>15}  {'Label':>10}  {'Conf':>8}  {'Δ Conf':>8}  Note")
print("-" * 65)
for word, note in targets:
    lbl, conf = predict(mask_word(sentence, word))
    print(f"{word:>15}  {lbl:>10}  {conf:>8.4f}  {conf - base_conf:>+8.4f}  {note}")
```

### Reflection Questions

**Q19.** Which masked word caused the largest drop in confidence? Did any masking cause the predicted label to flip? What do these results tell you about the relative causal importance of different tokens compared to what saliency scores alone predicted?

**Q20.** Masking "was" likely had little effect on the prediction, yet some function words may have had non-trivial saliency scores in Exercise 6. How do you reconcile these two findings? What does this suggest about the difference between *gradient sensitivity* and *causal importance*?

**Q21.** Consider the limitations of both saliency maps and perturbation tests as tools for understanding model behavior. What would a truly rigorous interpretability evaluation require, and why is NLP interpretability considered an open research problem rather than a solved one?

---

## Summary and Key Takeaways

This lab explored three complementary windows into how modern NLP models process language.

**Attention Mechanisms (Exercises 1–3)** revealed that transformers distribute focus across all tokens simultaneously, with different heads learning different linguistic patterns. Layer depth matters: early layers tend to capture local syntactic patterns while later layers encode longer-range semantic relationships. However, attention weights are a noisy proxy — a token attending to another does not guarantee the model has "understood" their relationship.

**Word Embeddings (Exercises 4–5)** demonstrated that meaning can be encoded geometrically: semantic categories cluster together, and relationships like gender or nationality become consistent directional offsets. This elegant structure also faithfully reflects human biases in training text, raising the important distinction between a model that has learned language statistics vs. one that has learned truth.

**Gradient Saliency and Perturbation (Exercises 6–7)** showed that we can interrogate model decisions by asking which inputs drive the output most strongly — and then test those claims empirically by removing tokens. The gap between correlational saliency and causal perturbation results is a recurring theme in interpretability research, reminding us that attribution is difficult even when we have full access to model internals.

Together, these exercises illustrate a key tension in modern AI: these systems are extraordinarily capable at language tasks, yet their internal representations remain only partially understood. Interpretability matters for trust, debugging, bias detection, and deploying AI responsibly.

---

## Submission Instructions

Create a new **public** Github Repository called `cs430`, upload your local `cs430` folder there including all code from this lab and:

Create `lab_nlp_results.md`:

```markdown
# Names: Your names here
# Lab: lab6 (NLP)
# Date: Today's date
```

And your answers to all reflection questions above. Each answer should be 2-5 sentences that demonstrate your understanding of the concepts through the lens of the exercises you ran.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the sixth lab!

---

## Optional Extensions

1. **Cross-sentence attention:** Feed BERT two sentences separated by `[SEP]` (a question and a passage). Using `bertviz`, explore whether attention crosses the sentence boundary — does this pattern change across layers?

2. **Swap the Winograd pronoun:** Modify the sentence to "…because **it** is too **small**" — now "it" should refer to "suitcase." Rerun Exercise 3 and compare the attention patterns. Does BERT update its coreference correctly?

3. **Adversarial perturbation:** In Exercise 7, replace sentiment words with near-synonyms of opposite valence ("brilliant" → "competent", "disaster" → "setback"). How much does this softer change affect confidence compared to masking?
