# Wordle Agent Strategies Lab
**Based on Game-Playing and Search Strategies from AI: A Modern Approach**

## Learning Objectives

By the end of this lab, you will be able to:

1. **Distinguish** between information-theoretic, minimax, and heuristic approaches to solving constraint satisfaction problems
2. **Compare** the computational trade-offs between optimal and approximate solution methods

## Lab Overview

This lab uses Wordle—the popular word-guessing game—as a testbed for comparing different AI agent architectures. Rather than implementing these agents yourself, you'll run complete implementations and observe how each approach makes decisions, handles uncertainty, and adapts its strategy based on feedback.

Wordle provides an excellent environment for studying AI decision-making because:
- The state space is large but manageable (12,972 possible five-letter words)
- Feedback is structured and deterministic (green/yellow/gray patterns)
- Success requires balancing information gathering with solution finding
- Different strategies reveal fundamental AI trade-offs

**Setup Requirements:**
- Python 3.10+
- Libraries: `collections`, `math`, `random`, `json`, `nltk`
- Optional: Ollama with `llama3.2` model for RL demonstrations
- Run once: `nltk.download('words')` to download the word corpus

Each exercise demonstrates a different agent strategy, allowing you to observe how theoretical concepts from search, game theory, and learning manifest in concrete decision-making.

---

I suggest running the following commands from your base user directory:

```bash
mkdir cs430 
cd cs430 
uv init 
uv add nltk
source .venv/bin/activate
touch wordle.ipynb
```

The last command will create a file such as `wordle.ipynb`. 

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


## Exercise 1: Understanding the Wordle Environment

### Description

Before comparing agent strategies, we need to understand the environment itself. This exercise implements the Wordle game mechanics and demonstrates how agents receive feedback. Understanding the state representation and feedback encoding is crucial—these design choices affect every agent strategy that follows.

### Key Concepts

- **State space**: The set of all possible game configurations (words that could be the answer)
- **Action space**: The set of valid guesses an agent can make
- **Feedback pattern**: A tuple encoding which letters are correct (green), present but misplaced (yellow), or absent (gray)
- **State transition function**: How the game state changes based on actions and observations
- **Constraint satisfaction**: Using feedback to eliminate impossible states

### Task

Run the code below and observe how Wordle feedback is generated. Pay attention to:
- How the pattern encoding handles repeated letters
- The order of operations (greens before yellows)
- How feedback constrains the remaining word space
- The difference between the answer word and the guess word

```python
from collections import Counter

def get_wordle_pattern(guess, answer):
    """
    Generate Wordle feedback pattern.
    Returns tuple: 0=gray (not in word), 1=yellow (wrong position), 2=green (correct)
    """
    result = [0] * 5
    answer_counts = Counter(answer)
    
    # First pass: mark all exact matches (green)
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = 2
            answer_counts[g] -= 1  # Consume this letter
    
    # Second pass: mark letters in wrong positions (yellow)
    for i, g in enumerate(guess):
        if result[i] == 0 and answer_counts[g] > 0:
            result[i] = 1
            answer_counts[g] -= 1  # Consume this letter
    
    return tuple(result)

def apply_constraints(word, guess, pattern):
    """Check if a word is consistent with observed guess/pattern"""
    return get_wordle_pattern(guess, word) == pattern

# Load word list from NLTK
print("=== Loading Word List ===\n")

try:
    import nltk
    from nltk.corpus import words
    
    # Download if not already present
    try:
        words.words()
    except LookupError:
        print("Downloading NLTK words corpus...")
        nltk.download('words', quiet=True)
    
    # Get all 5-letter words, uppercase
    all_words = words.words()
    WORD_LIST = sorted(set(word.upper() for word in all_words if len(word) == 5 and word.isalpha()))
    
    print(f"Loaded {len(WORD_LIST)} five-letter words from NLTK")
    print(f"Sample words: {WORD_LIST[:10]}")
    print()
    
except ImportError:
    print("NLTK not installed. Using fallback word list.")
    # Fallback to a larger curated list
    WORD_LIST = ["SALET", "ROATE", "RAISE", "ARISE", "IRATE", "CRANE", "SLATE", 
                 "CRATE", "TRACE", "STARE", "ADORE", "ALONE", "ATONE", "STORE",
                 "SHORE", "SNORE", "SPORE", "SCORE", "SWORE", "SHONE", "STONE",
                 "DRONE", "PRONE", "OZONE", "PHONE", "THOSE", "WHOSE", "CHOSE",
                 "BRAKE", "FLAKE", "SHAKE", "SNAKE", "QUAKE", "AWAKE"]
    print(f"Using fallback list: {len(WORD_LIST)} words\n")

# Demonstration
print("=== Wordle Feedback Mechanism ===\n")

test_cases = [
    ("ROBOT", "FLOOR"),  # Repeated letters
    ("CRANE", "REACT"),  # Anagrams
    ("SALET", "LASER"),  # Close match
]

for guess, answer in test_cases:
    pattern = get_wordle_pattern(guess, answer)
    pattern_str = ''.join(['⬜' if p == 0 else '🟨' if p == 1 else '🟩' for p in pattern])
    print(f"Guess: {guess}")
    print(f"Answer: {answer}")
    print(f"Pattern: {pattern_str} {pattern}")
    print()

# Show constraint propagation with realistic word list
print("=== Constraint Propagation Example ===\n")
# Select words starting with 'FL' from full list
sample_words = [w for w in WORD_LIST if w.startswith('FL')][:8]
if len(sample_words) < 5:
    sample_words = WORD_LIST[:8]

guess = "CRANE"
answer = sample_words[0] if sample_words else "FLOUR"
pattern = get_wordle_pattern(guess, answer)

print(f"Sample from {len(WORD_LIST)} total words: {sample_words}")
print(f"After guessing '{guess}' with answer '{answer}':")
print(f"Pattern: {pattern}")
remaining = [w for w in sample_words if apply_constraints(w, guess, pattern)]
print(f"Remaining possibilities: {remaining}")
print(f"Eliminated: {set(sample_words) - set(remaining)}")
print(f"\nFrom full word list, {len([w for w in WORD_LIST if apply_constraints(w, guess, pattern)])} words remain")
```

### Reflection Questions

1. In the constraint propagation example, explain the relationship between the pattern encoding and the reduction in state space. How does each piece of information (gray, yellow, green) contribute differently to narrowing possibilities?

2. The feedback mechanism is deterministic—the same guess against the same answer always produces the same pattern. How does this property affect the agent strategies we'll explore? Would probabilistic feedback change the problem fundamentally?

---

## Exercise 2: Frequency-Based Heuristic Agent

### Description

The simplest approach uses letter frequency heuristics: prioritize words containing common letters in common positions. This agent doesn't reason about information gain—it simply uses statistical patterns from the English language. While fast and intuitive, this "greedy" approach may not be optimal.

### Key Concepts

- **Heuristic function**: A rule of thumb that estimates the value of a decision without exhaustive search
- **Letter frequency distribution**: Statistical analysis of how often letters appear in valid words
- **Positional frequency**: How often specific letters appear in specific positions
- **Greedy strategy**: Making locally optimal choices at each step
- **Unique letter bonus**: Preferring words with more distinct letters to gather broader information

### Task

Run this frequency-based agent and observe its behavior. Notice:
- Which words it prefers as opening guesses
- How it handles the trade-off between common letters and unique letters
- Whether it adapts its strategy as the game progresses
- How quickly it narrows down possibilities

```python
from collections import defaultdict, Counter

# Letter frequency data from Wikipedia (English text)
# https://en.wikipedia.org/wiki/Letter_frequency
LETTER_FREQUENCIES = {
    'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97,
    'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
    'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
    'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
    'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10,
    'Z': 0.07
}

def build_frequency_tables(words):
    """Calculate letter frequencies by position from word list"""
    position_freq = [defaultdict(int) for _ in range(5)]
    
    for word in words:
        for i, letter in enumerate(word):
            position_freq[i][letter] += 1
    
    return position_freq

def score_word_frequency(word, freq_tables, remaining_words):
    """Score word based on letter/position frequencies"""
    # Use position-specific frequencies from the remaining words
    score = sum(freq_tables[i][letter] for i, letter in enumerate(word))
    
    # Bonus for unique letters (more information gathered)
    unique_ratio = len(set(word)) / len(word)
    score *= (1 + unique_ratio)
    
    # Add bonus based on general English letter frequency
    english_freq_score = sum(LETTER_FREQUENCIES.get(letter, 0) for letter in set(word))
    score += english_freq_score * 0.1  # Small weight for general frequency
    
    return score

def frequency_agent_solve(answer, word_list, verbose=True):
    """Solve Wordle using frequency heuristics"""
    remaining = word_list.copy()
    guesses = []
    
    for attempt in range(6):
        freq_tables = build_frequency_tables(remaining)
        
        # Choose word with highest frequency score from remaining possibilities
        best_word = max(remaining, key=lambda w: score_word_frequency(w, freq_tables, remaining))
        guesses.append(best_word)
        
        pattern = get_wordle_pattern(best_word, answer)
        
        if verbose:
            pattern_str = ''.join(['⬜' if p == 0 else '🟨' if p == 1 else '🟩' for p in pattern])
            print(f"Attempt {attempt + 1}: {best_word} -> {pattern_str}")
            print(f"  Remaining words: {len(remaining)}")
        
        if best_word == answer:
            if verbose:
                print(f"✓ Solved in {len(guesses)} guesses!\n")
            return guesses
        
        # Update remaining words based on constraints
        remaining = [w for w in remaining if apply_constraints(w, best_word, pattern)]
        
        if not remaining:
            if verbose:
                print(f"✗ No valid words remain! Answer was {answer}\n")
            return guesses
    
    if verbose:
        print(f"✗ Failed to solve in 6 guesses. Answer was {answer}\n")
    return guesses

# Test the frequency agent
print("=== Frequency-Based Heuristic Agent ===\n")
print(f"Total word list size: {len(WORD_LIST)}\n")

# Select diverse test words
import random
random.seed(42)
test_answers = random.sample(WORD_LIST, 3)

print(f"Testing on random words: {test_answers}\n")

for answer in test_answers:
    print(f"Target word: {answer}")
    result = frequency_agent_solve(answer, WORD_LIST)
    print()

# Show top words by frequency
print("=== Analysis: Best Opening Words by Frequency ===\n")
freq_tables = build_frequency_tables(WORD_LIST)
all_scores = [(w, score_word_frequency(w, freq_tables, WORD_LIST)) for w in WORD_LIST]
all_scores.sort(key=lambda x: x[1], reverse=True)

print("Top 10 words by frequency score:")
for i, (word, score) in enumerate(all_scores[:10], 1):
    unique_letters = len(set(word))
    print(f"{i:2}. {word}: {score:.2f} ({unique_letters} unique letters)")
```

### Reflection Questions

3. The frequency agent uses a "greedy" approach—it always picks what seems best locally without considering future moves. Describe a scenario where this could lead to suboptimal performance compared to an agent that thinks ahead.

4. Notice how the unique letter bonus affects word selection. Why might choosing words with five distinct letters be advantageous in early guesses but potentially wasteful in later guesses?

5. This agent recalculates frequencies based on the remaining word list after each guess. How does this adaptive behavior differ from using fixed, pre-computed frequencies? What are the computational trade-offs?

---

## Exercise 3: Information Theory Agent (Entropy Maximization)

### Description

Information theory provides a mathematically principled approach: choose guesses that maximize expected information gain. This agent calculates entropy—a measure of uncertainty—and selects words that reduce uncertainty most effectively. This is the theoretically optimal strategy for minimizing average guesses.

### Key Concepts

- **Entropy**: A measure of uncertainty or randomness in a probability distribution (higher entropy = more uncertainty)
- **Information gain**: The reduction in entropy achieved by making an observation
- **Expected information**: The average information gained across all possible outcomes, weighted by probability
- **Partition**: How a guess divides the remaining possibilities into groups based on pattern feedback
- **Optimization criterion**: Maximizing expected information rather than minimizing worst case

### Task

Run the information theory agent and observe:
- How entropy calculations influence word choice
- The difference between entropy-optimal guesses and frequency-based guesses
- How quickly this agent converges on answers
- The computational cost compared to the frequency heuristic

```python
import math

def calculate_pattern_entropy(word, possible_words):
    """
    Calculate expected information gain (entropy) for a guess.
    Higher entropy = more expected information gained.
    """
    if not possible_words:
        return 0
    
    # Count how many words produce each pattern
    pattern_counts = {}
    for candidate in possible_words:
        pattern = get_wordle_pattern(word, candidate)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Calculate entropy: -sum(p * log2(p))
    total = len(possible_words)
    entropy = 0
    for count in pattern_counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * math.log2(probability)
    
    return entropy

def entropy_agent_solve(answer, word_list, verbose=True):
    """Solve Wordle using information theory (entropy maximization)"""
    remaining = word_list.copy()
    guesses = []
    
    for attempt in range(6):
        # For first guess, we can pre-compute on full list
        # For subsequent guesses, only consider remaining words for efficiency
        if attempt == 0 and len(remaining) > 1000:
            # Sample for efficiency on first guess with large word list
            sample_size = min(2000, len(remaining))
            candidates = random.sample(remaining, sample_size)
        else:
            candidates = remaining
        
        # Calculate entropy for candidate guesses
        word_scores = [(w, calculate_pattern_entropy(w, remaining)) for w in candidates]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_word = word_scores[0][0]
        best_entropy = word_scores[0][1]
        guesses.append(best_word)
        
        pattern = get_wordle_pattern(best_word, answer)
        
        if verbose:
            pattern_str = ''.join(['⬜' if p == 0 else '🟨' if p == 1 else '🟩' for p in pattern])
            print(f"Attempt {attempt + 1}: {best_word} -> {pattern_str}")
            print(f"  Entropy: {best_entropy:.2f} bits")
            print(f"  Remaining words: {len(remaining)}")
        
        if best_word == answer:
            if verbose:
                print(f"✓ Solved in {len(guesses)} guesses!\n")
            return guesses
        
        remaining = [w for w in remaining if apply_constraints(w, best_word, pattern)]
        
        if not remaining:
            if verbose:
                print(f"✗ No valid words remain! Answer was {answer}\n")
            return guesses
    
    if verbose:
        print(f"✗ Failed to solve in 6 guesses. Answer was {answer}\n")
    return guesses

# Test the entropy agent
print("=== Information Theory (Entropy) Agent ===\n")

# Use same test words for comparison
random.seed(42)
test_answers = random.sample(WORD_LIST, 3)

print(f"Testing on: {test_answers}\n")

for answer in test_answers:
    print(f"Target word: {answer}")
    result = entropy_agent_solve(answer, WORD_LIST)
    print()

# Compare first guesses
print("=== First Guess Comparison ===\n")

# Sample words for entropy calculation (full list too slow)
print("Calculating best opening words (sampling for efficiency)...\n")
sample_words = random.sample(WORD_LIST, min(1000, len(WORD_LIST)))

print("Top 5 words by frequency score:")
freq_tables = build_frequency_tables(WORD_LIST)
freq_scores = [(w, score_word_frequency(w, freq_tables, WORD_LIST)) for w in sample_words]
freq_scores.sort(key=lambda x: x[1], reverse=True)
for word, score in freq_scores[:5]:
    print(f"  {word}: {score:.2f}")

print("\nTop 5 words by entropy:")
entropy_scores = [(w, calculate_pattern_entropy(w, WORD_LIST)) for w in sample_words]
entropy_scores.sort(key=lambda x: x[1], reverse=True)
for word, entropy in entropy_scores[:5]:
    print(f"  {word}: {entropy:.2f} bits")
```

### Reflection Questions

6. Entropy measures how evenly a guess partitions the remaining possibilities. Explain why a guess that splits possibilities into equally-sized groups has higher entropy than one that creates very uneven partitions. Use a concrete example from the output.

7. The entropy agent might choose a word that cannot possibly be the answer if it maximizes information gain. Is this rational? Discuss the trade-off between "playing to win immediately" versus "playing to gather information."

8. Compare the computational complexity of the frequency heuristic versus entropy calculation. As the word list grows larger, which approach scales better and why? What does this tell us about the trade-off between optimality and feasibility?

---

## Exercise 4: Minimax Agent (Worst-Case Optimization)

### Description

While entropy optimizes average performance, the minimax strategy optimizes worst-case performance. This agent selects guesses that minimize the maximum number of remaining possibilities—guaranteeing the best worst-case outcome. This reflects a fundamentally different risk profile than the entropy approach.

### Key Concepts

- **Minimax principle**: Choose actions that minimize the worst possible outcome
- **Worst-case analysis**: Evaluating strategies based on their performance in adversarial scenarios
- **Max partition size**: The largest group of words that share the same pattern feedback
- **Guaranteed bounds**: Upper limits on performance regardless of which answer is chosen
- **Risk aversion**: Preferring strategies with better worst-case guarantees over better average cases

### Task

Run the minimax agent and observe:
- How it differs from entropy in word selection
- Its behavior when possibilities narrow down
- The guaranteed bounds it provides
- Situations where it outperforms or underperforms entropy

```python
def calculate_max_partition(word, possible_words):
    """
    Calculate the size of the largest partition (worst case).
    Lower is better - means we're guaranteed to eliminate more words.
    """
    if not possible_words:
        return 0
    
    pattern_counts = {}
    for candidate in possible_words:
        pattern = get_wordle_pattern(word, candidate)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Return size of largest partition
    return max(pattern_counts.values()) if pattern_counts else 0

def minimax_agent_solve(answer, word_list, verbose=True):
    """Solve Wordle using minimax (minimize worst-case remaining)"""
    remaining = word_list.copy()
    guesses = []
    
    for attempt in range(6):
        # Sample for efficiency on large word lists
        if len(remaining) > 500:
            candidates = random.sample(remaining, min(500, len(remaining)))
        else:
            candidates = remaining
        
        # Find word that minimizes maximum partition size
        word_scores = [(w, calculate_max_partition(w, remaining)) for w in candidates]
        word_scores.sort(key=lambda x: x[1])  # Lower is better
        
        best_word = word_scores[0][0]
        worst_case = word_scores[0][1]
        guesses.append(best_word)
        
        pattern = get_wordle_pattern(best_word, answer)
        
        if verbose:
            pattern_str = ''.join(['⬜' if p == 0 else '🟨' if p == 1 else '🟩' for p in pattern])
            print(f"Attempt {attempt + 1}: {best_word} -> {pattern_str}")
            print(f"  Worst-case remaining: {worst_case}")
            print(f"  Actual remaining: {len(remaining)}")
        
        if best_word == answer:
            if verbose:
                print(f"✓ Solved in {len(guesses)} guesses!\n")
            return guesses
        
        remaining = [w for w in remaining if apply_constraints(w, best_word, pattern)]
        
        if not remaining:
            if verbose:
                print(f"✗ No valid words remain! Answer was {answer}\n")
            return guesses
    
    if verbose:
        print(f"✗ Failed to solve in 6 guesses. Answer was {answer}\n")
    return guesses

# Test minimax agent
print("=== Minimax (Worst-Case Optimization) Agent ===\n")

random.seed(42)
test_answers = random.sample(WORD_LIST, 3)

print(f"Testing on: {test_answers}\n")

for answer in test_answers:
    print(f"Target word: {answer}")
    result = minimax_agent_solve(answer, WORD_LIST)
    print()

# Strategy comparison on first guess
print("=== Strategy Comparison for First Guess ===\n")
print("Analyzing sample words...\n")

sample_words = random.sample(WORD_LIST, min(200, len(WORD_LIST)))

# Get top words by each metric
entropy_ranked = sorted([(w, calculate_pattern_entropy(w, WORD_LIST)) for w in sample_words],
                        key=lambda x: x[1], reverse=True)
minimax_ranked = sorted([(w, calculate_max_partition(w, WORD_LIST)) for w in sample_words],
                        key=lambda x: x[1])

print("Top 5 by Entropy (maximize information):")
for word, entropy in entropy_ranked[:5]:
    max_part = calculate_max_partition(word, WORD_LIST)
    print(f"  {word}: {entropy:.2f} bits (max partition: {max_part})")

print("\nTop 5 by Minimax (minimize worst case):")
for word, max_part in minimax_ranked[:5]:
    entropy = calculate_pattern_entropy(word, WORD_LIST)
    print(f"  {word}: max partition {max_part} (entropy: {entropy:.2f} bits)")
```

### Reflection Questions

9. The minimax agent provides a guarantee: "No matter what the answer is, I'll have at most X words remaining after this guess." Why might this guarantee be valuable in competitive or time-limited scenarios even if average performance is worse?

10. Observe cases where entropy and minimax choose different first words. Describe the philosophical difference in these strategies: one is optimistic (optimizing average case) and one is pessimistic (optimizing worst case). Which worldview seems more appropriate for Wordle, and why?

11. As the number of remaining possibilities decreases (e.g., down to 2-3 words), do you notice the strategies converging? Explain why different optimization criteria matter more when uncertainty is high versus low.

---

## Exercise 5: Hybrid Strategy Agent

### Description

Real-world agents often combine multiple strategies. This hybrid agent uses entropy maximization early (when information gathering matters most) but switches to minimax near the end (when guaranteeing success matters more). This demonstrates how agents can adaptively change strategies based on game state.

### Key Concepts

- **Adaptive strategy**: Changing decision-making approach based on current state
- **Phase transition**: Switching between strategies at critical thresholds
- **Strategy composition**: Combining multiple approaches within a single agent
- **Threshold-based switching**: Using game state features to trigger strategy changes
- **Meta-strategy**: A strategy for choosing strategies

### Task

Run the hybrid agent and observe:
- When it switches from entropy to minimax
- How the switch point affects performance
- Whether hybrid performance exceeds either pure strategy
- The computational cost of running both calculations

```python
def hybrid_agent_solve(answer, word_list, switch_threshold=5, verbose=True):
    """
    Hybrid strategy: use entropy when many possibilities remain,
    switch to minimax when few remain.
    """
    remaining = word_list.copy()
    guesses = []
    
    for attempt in range(6):
        # Choose strategy based on remaining possibilities
        if len(remaining) > switch_threshold:
            # Use entropy for information gathering
            word_scores = [(w, calculate_pattern_entropy(w, remaining)) for w in remaining]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            strategy_used = "ENTROPY"
        else:
            # Use minimax for guaranteed bounds
            word_scores = [(w, calculate_max_partition(w, remaining)) for w in remaining]
            word_scores.sort(key=lambda x: x[1])  # Lower is better
            strategy_used = "MINIMAX"
        
        best_word = word_scores[0][0]
        guesses.append(best_word)
        
        pattern = get_wordle_pattern(best_word, answer)
        
        if verbose:
            pattern_str = ''.join(['⬜' if p == 0 else '🟨' if p == 1 else '🟩' for p in pattern])
            print(f"Attempt {attempt + 1}: {best_word} -> {pattern_str} [{strategy_used}]")
            print(f"  Remaining words: {len(remaining)}")
        
        if best_word == answer:
            if verbose:
                print(f"✓ Solved in {len(guesses)} guesses!\n")
            return guesses
        
        remaining = [w for w in remaining if apply_constraints(w, best_word, pattern)]
    
    if verbose:
        print(f"✗ Failed to solve. Answer was {answer}\n")
    return guesses

# Test hybrid agent
print("=== Hybrid Strategy Agent ===\n")
print("Strategy: Entropy when >5 words remain, Minimax otherwise\n")

test_answers = ["DOWEL", "GNOME", "FUGUE"]
for answer in test_answers:
    print(f"Target word: {answer}")
    result = hybrid_agent_solve(answer, WORD_LIST, switch_threshold=5)
    print()
```

### Reflection Questions

12. The hybrid agent switches strategies mid-game. Describe what fundamentally changes about the problem when you have 10 remaining possibilities versus 2 remaining possibilities that justifies this strategy change.

13. The threshold parameter (when to switch strategies) is somewhat arbitrary. Propose a method for automatically determining the optimal threshold based on the statistical properties of the word list. What factors should influence this decision?

14. Could we extend this hybrid approach to three or more strategies? Sketch out a meta-strategy that might incorporate frequency heuristics, entropy, and minimax at different stages. What would be the benefits and costs?

---

## Exercise 6: Performance Benchmarking Across Strategies

### Description

To scientifically compare strategies, we need systematic benchmarking. This exercise runs all agents against the same set of target words and collects performance metrics. Observing these results reveals the practical differences between theoretical optimality and real-world performance.

### Task

Run the benchmark and observe:
- Which strategy has the best average performance
- Which has the best worst-case performance
- The variance in performance across different strategies
- Computational time differences (if running on larger word lists)

```python
import random
from time import time

def run_benchmark(word_list, num_trials=10):
    """Compare all strategies on random sample of words"""
    
    # Select random target words
    random.seed(42)  # For reproducibility
    test_words = random.sample(word_list, min(num_trials, len(word_list)))
    
    agents = {
        'Frequency': frequency_agent_solve,
        'Entropy': entropy_agent_solve,
        'Minimax': minimax_agent_solve,
        'Hybrid': lambda ans, wl, v: hybrid_agent_solve(ans, wl, 50, v)
    }
    
    results = {name: [] for name in agents}
    
    print("=== Performance Benchmark ===\n")
    print(f"Testing {len(test_words)} random words from {len(word_list)} total")
    print(f"Test words: {test_words}\n")
    
    for name, agent_func in agents.items():
        print(f"Testing {name} agent...")
        start_time = time()
        
        for word in test_words:
            guesses = agent_func(word, word_list, verbose=False)
            results[name].append(len(guesses))
        
        elapsed = time() - start_time
        
        avg_guesses = sum(results[name]) / len(results[name])
        max_guesses = max(results[name])
        min_guesses = min(results[name])
        
        print(f"  Average: {avg_guesses:.2f} guesses")
        print(f"  Best: {min_guesses}, Worst: {max_guesses}")
        print(f"  Time: {elapsed:.3f}s")
        print()
    
    # Summary comparison
    print("=== Summary Comparison ===\n")
    print(f"{'Strategy':<12} {'Avg':<6} {'Min':<5} {'Max':<5} {'Success%':<10}")
    print("-" * 45)
    for name in agents:
        avg = sum(results[name]) / len(results[name])
        success_rate = sum(1 for g in results[name] if g <= 6) / len(results[name]) * 100
        print(f"{name:<12} {avg:<6.2f} {min(results[name]):<5} {max(results[name]):<5} {success_rate:<10.1f}")
    
    return results

# Run benchmark
print("Note: With a large word list, this may take a minute...\n")
results = run_benchmark(WORD_LIST, num_trials=2)

# Additional analysis
print("\n=== Word List Statistics ===\n")
print(f"Total words: {len(WORD_LIST)}")
print(f"Words with 5 unique letters: {sum(1 for w in WORD_LIST if len(set(w)) == 5)}")
print(f"Words with repeated letters: {sum(1 for w in WORD_LIST if len(set(w)) < 5)}")

# Letter distribution
all_letters = Counter()
for word in WORD_LIST:
    all_letters.update(word)

print(f"\nMost common letters in word list:")
for letter, count in all_letters.most_common(10):
    percentage = (count / (len(WORD_LIST) * 5)) * 100
    wiki_freq = LETTER_FREQUENCIES.get(letter, 0)
    print(f"  {letter}: {count:5} ({percentage:5.2f}% vs {wiki_freq:5.2f}% in English text)")
```

### Reflection Questions

15. Examine the average versus worst-case performance across strategies. Does the "best on average" strategy also have the "best worst case"? Discuss why optimization criteria matter when choosing between algorithms.

16. The benchmark uses a random sample of words. How might results differ if we tested on: (a) only common words, (b) words with unusual letter patterns, or (c) the full 12,000+ word dictionary? What does this reveal about the generalizability of our findings?

17. Notice the computational time differences between strategies. In a real-time game where players have limited thinking time, how should we balance solution quality against computation speed? Propose a practical decision rule.

---

## Exercise 7: Simulated Reinforcement Learning Agent

### Description

Unlike the previous rule-based agents, a reinforcement learning (RL) agent learns strategy through experience. Since training an RL agent from scratch is computationally intensive, we'll simulate RL decision-making using a language model (Ollama with llama3.2) that can reason about Wordle strategy. This demonstrates how modern AI can learn implicit strategies without explicit programming.

### Key Concepts

- **Reinforcement learning**: Learning optimal behavior through trial-and-error interaction with an environment
- **State-action-reward**: The fundamental RL loop where agents learn which actions yield best outcomes
- **Policy**: A strategy mapping states to actions (what the agent has learned)
- **Exploration vs exploitation**: Balancing trying new approaches versus using known good strategies
- **Emergent behavior**: Complex strategies that arise from simple learning rules

### Task

**Note**: This requires Ollama installed with `llama3.2` model. If unavailable, read through the code to understand the approach.

Run the RL simulation and observe:
- What strategy the model develops
- How it reasons about Wordle gameplay
- Whether it discovers principles similar to our algorithmic agents
- The quality of its explanations

```python
# Simulated RL agent using Ollama
# Requires: pip install ollama (or just observe the code structure)

def simulate_rl_agent_with_llm(answer, word_list, verbose=True):
    """
    Simulate an RL agent using an LLM to make strategic decisions.
    The LLM has 'learned' Wordle strategy from its training data.
    """
    try:
        import ollama
    except ImportError:
        print("Ollama not installed. Showing simulation structure only.\n")
        return []
    
    remaining = word_list.copy()
    guesses = []
    history = []
    
    if verbose:
        print("=== Simulated RL Agent (via llama3.2) ===\n")
    
    for attempt in range(6):
        # Build prompt with current game state
        remaining_sample = remaining[:20] if len(remaining) > 20 else remaining
        prompt = f"""You are playing Wordle. Based on your experience:

Current state:
- Attempt {attempt + 1} of 6
- Remaining possible words ({len(remaining)}): {remaining_sample}
- Previous guesses and patterns: {history if history else 'None yet'}

You MUST choose ONE word from the remaining possibilities list above.
Respond with ONLY the chosen word in uppercase, nothing else.
"""
        
        try:
            client = ollama.Client(host='http://ollama.cs.wallawalla.edu:11434')
            response = client.generate(model='llama3.2', prompt=prompt)
            full_response = response['response'].strip()
            
            if verbose:
                print(f"LLM response: {full_response[:150]}...")
            
            # Try to extract a valid word from response
            # Check each line and each word in the response
            chosen_word = None
            for line in full_response.upper().split('\n'):
                # Remove punctuation and get words
                words = line.replace(',', ' ').replace('.', ' ').split()
                for word in words:
                    clean_word = ''.join(c for c in word if c.isalpha())
                    if clean_word in remaining:
                        chosen_word = clean_word
                        break
                if chosen_word:
                    break
            
            # Fallback: use highest entropy word from remaining
            if not chosen_word:
                if verbose:
                    print(f"  LLM output '{full_response[:50]}...' not in remaining words, using fallback...")
                # Use entropy to pick a good fallback
                if len(remaining) <= 50:
                    word_scores = [(w, calculate_pattern_entropy(w, remaining)) for w in remaining]
                    word_scores.sort(key=lambda x: x[1], reverse=True)
                    chosen_word = word_scores[0][0]
                else:
                    chosen_word = remaining[0]
            
            guesses.append(chosen_word)
            pattern = get_wordle_pattern(chosen_word, answer)
            
            if verbose:
                pattern_str = ''.join(['⬜' if p == 0 else '🟨' if p == 1 else '🟩' for p in pattern])
                print(f"\nAttempt {attempt + 1}: {chosen_word} -> {pattern_str}")
                print(f"Remaining: {len(remaining)}\n")
            
            if chosen_word == answer:
                if verbose:
                    print(f"✓ Solved in {len(guesses)} guesses!\n")
                return guesses
            
            history.append(f"{chosen_word} -> {pattern}")
            remaining = [w for w in remaining if apply_constraints(w, chosen_word, pattern)]
            
            if not remaining:
                if verbose:
                    print(f"✗ No valid words remain! Answer was {answer}\n")
                return guesses
            
        except Exception as e:
            if verbose:
                print(f"Error with Ollama: {e}")
                print(f"Falling back to entropy strategy...\n")
            # Fallback to entropy
            if remaining:
                if len(remaining) <= 50:
                    word_scores = [(w, calculate_pattern_entropy(w, remaining)) for w in remaining]
                    word_scores.sort(key=lambda x: x[1], reverse=True)
                    chosen_word = word_scores[0][0]
                else:
                    chosen_word = remaining[0]
                
                guesses.append(chosen_word)
                pattern = get_wordle_pattern(chosen_word, answer)
                
                if chosen_word == answer:
                    if verbose:
                        print(f"✓ Solved in {len(guesses)} guesses!\n")
                    return guesses
                
                remaining = [w for w in remaining if apply_constraints(w, chosen_word, pattern)]
    
    if verbose:
        print(f"✗ Failed to solve. Answer was {answer}\n")
    return guesses

# Demonstrate RL concept (with or without actual Ollama)
print("=== Reinforcement Learning Approach ===\n")
print("RL agents learn through experience rather than explicit rules.")
print("They discover strategies by:")
print("1. Trying actions in various states")
print("2. Receiving rewards (solved in fewer guesses = better)")
print("3. Updating their policy to prefer successful actions")
print("4. Balancing exploration (trying new things) vs exploitation (using what works)\n")

print("Modern LLMs like llama3.2 have implicitly 'learned' Wordle strategies")
print("from seeing many examples during training.\n")

# Try to run actual RL simulation if Ollama available
try:
    import ollama
    print("Ollama detected! Testing LLM-based agent...\n")
    
    # Use a small subset for LLM testing (faster and more reliable)
    random.seed(789)
    llm_word_list = random.sample(WORD_LIST, min(100, len(WORD_LIST)))
    test_answer = random.choice(llm_word_list)
    
    print(f"Testing with word list of {len(llm_word_list)} words")
    print(f"Target: {test_answer}\n")
    
    result = simulate_rl_agent_with_llm(test_answer, llm_word_list, verbose=True)
    
    if result:
        print(f"LLM agent completed in {len(result)} guesses: {' -> '.join(result)}")
    
except ImportError:
    print("Ollama not available - showing conceptual simulation only.")
    print("Install with: pip install ollama")
    print("Then run: ollama pull llama3.2\n")
except Exception as e:
    print(f"Could not connect to Ollama: {e}")
    print("Make sure Ollama is running with: ollama serve\n")
```

### Reflection Questions

18. Traditional RL requires millions of training episodes to learn Wordle strategy. Modern LLMs appear to play reasonably well without explicit training on Wordle. What does this suggest about how these models represent and transfer knowledge across different tasks?

19. Compare the transparency of the rule-based agents (entropy, minimax) versus the RL/LLM agent. Which approach makes it easier to understand *why* a particular guess was chosen? Discuss the trade-off between interpretability and performance.

20. The RL agent needs to balance exploration (trying suboptimal guesses to learn) versus exploitation (using its current best strategy). How do the deterministic agents we built earlier handle this trade-off implicitly through their optimization criteria?

---

## Exercise 8: Strategic Insights and Failure Analysis

### Description

Our final exercise examines edge cases and failure modes. By intentionally creating difficult scenarios, we can understand the limitations of each approach and identify when theoretical optimality diverges from practical performance.

### Key Concepts

- **Adversarial examples**: Inputs specifically designed to challenge an algorithm
- **Failure mode analysis**: Systematic study of when and why algorithms fail
- **Robustness**: How well a strategy performs on difficult or unusual inputs
- **Degeneracy**: Cases where multiple strategies perform identically
- **Domain assumptions**: Implicit constraints that algorithms rely on

### Task

Run the failure analysis and observe:
- Which types of words are hardest for each strategy
- Whether any strategy consistently handles difficult cases better
- Patterns in failures (e.g., repeated letters, uncommon patterns)
- How "unlucky" word choices affect different agents

```python
def analyze_difficult_cases(word_list):
    """Identify and test edge cases that challenge agents"""
    
    print("=== Failure Mode Analysis ===\n")
    
    # Case 1: Words with repeated letters
    repeated_letter_words = [w for w in word_list if len(set(w)) < 5]
    
    print(f"Case 1: Words with Repeated Letters")
    print(f"(Challenge: Less information per guess)\n")
    print(f"Found {len(repeated_letter_words)} words with repeated letters")
    
    if repeated_letter_words:
        # Test a few diverse examples
        test_words = random.sample(repeated_letter_words, min(3, len(repeated_letter_words)))
        print(f"Testing: {test_words}\n")
        
        agents = {
            'Frequency': frequency_agent_solve,
            'Entropy': entropy_agent_solve,
            'Minimax': minimax_agent_solve,
        }
        
        for test_word in test_words:
            unique_count = len(set(test_word))
            print(f"{test_word} ({unique_count} unique letters):")
            for name, agent in agents.items():
                guesses = agent(test_word, word_list, verbose=False)
                print(f"  {name}: {len(guesses)} guesses")
            print()
    
    # Case 2: Similar words (share many letters)
    print("\nCase 2: Word Clusters (Minimal Distinguishing Features)")
    print("(Challenge: Many words fit the same pattern)\n")
    
    # Find words with common prefixes/suffixes
    prefixes = defaultdict(list)
    for word in word_list:
        if len(word) >= 3:
            prefixes[word[:3]].append(word)
    
    large_clusters = [(prefix, words) for prefix, words in prefixes.items() if len(words) >= 5]
    large_clusters.sort(key=lambda x: len(x[1]), reverse=True)
    
    if large_clusters:
        prefix, cluster = large_clusters[0]
        print(f"Largest cluster with prefix '{prefix}': {len(cluster)} words")
        print(f"Sample: {cluster[:8]}\n")
        
        # Test on a couple from the cluster
        test_cluster = cluster[:3]
        for target in test_cluster:
            print(f"Target: {target}")
            for name, agent in {'Entropy': entropy_agent_solve,
                                'Minimax': minimax_agent_solve}.items():
                guesses = agent(target, word_list, verbose=False)
                print(f"  {name}: {len(guesses)} guesses")
            print()
    
    # Case 3: Uncommon letters
    print("\nCase 3: Words with Uncommon Letters")
    print("(Challenge: Less guided by letter frequency)\n")
    
    uncommon_letters = ['Q', 'X', 'Z', 'J']
    uncommon_words = [w for w in word_list if any(letter in w for letter in uncommon_letters)]
    
    if uncommon_words:
        print(f"Found {len(uncommon_words)} words with Q/X/Z/J")
        test_uncommon = random.sample(uncommon_words, min(3, len(uncommon_words)))
        print(f"Testing: {test_uncommon}\n")
        
        for target in test_uncommon:
            uncommon_in_word = [l for l in target if l in uncommon_letters]
            print(f"{target} (contains: {uncommon_in_word}):")
            for name in ['Frequency', 'Entropy']:
                agent = agents.get(name) or frequency_agent_solve if name == 'Frequency' else entropy_agent_solve
                guesses = agent(target, word_list, verbose=False)
                print(f"  {name}: {len(guesses)} guesses")
            print()

# Run failure analysis
print("Analyzing edge cases from full word list...\n")
analyze_difficult_cases(WORD_LIST)
```

### Reflection Questions

21. Words with repeated letters (like "ATONE" with two instances of a vowel) challenge our agents differently. Explain why repeated letters reduce the information gained per guess and how this affects entropy calculations.

22. When multiple words remain that differ by only one letter (like "STARE", "SCARE", "SPARE"), the game becomes partially luck-based. Discuss how each strategy (frequency, entropy, minimax) handles this degeneracy and whether any approach is fundamentally superior in these cases.

23. Consider the broader implications: Wordle has a finite, known state space (12,000+ words). How would these strategies need to adapt for a game with an unknown or infinite state space (like Scrabble or real-world decision problems)? What assumptions would break down?

---

## Summary and Key Takeaways

Through these exercises, you've observed how different AI paradigms approach the same problem:

**Core Strategic Insights:**
- **Heuristics** provide fast, intuitive solutions but may miss optimal strategies
- **Information theory** optimizes average performance through systematic uncertainty reduction
- **Minimax** guarantees bounds by optimizing worst-case scenarios
- **Hybrid approaches** combine strengths by adapting to game state
- **Learning-based methods** can discover strategies without explicit programming

---

## Submission Instructions

Create a new **public** Github Repository called `cs430`, upload your local `cs430` folder there including all code from this lab and:

Create `lab_wordle_results.md`:

```markdown
# Names: Your names here
# Lab: lab3 (Wordle)
# Date: Today's date
```

And your answers to all reflection questions above. Each answer should be 2-5 sentences that demonstrate your understanding of the concepts through the lens of the exercises you ran.

Email the GitHub repository web link to me at `chike.abuah@wallawalla.edu`

*If you're concerned about privacy* 

You can make a **private** Github Repo and add me as a collaborator, my username is `abuach`.

Congrats, you're done with the fifth lab!

---