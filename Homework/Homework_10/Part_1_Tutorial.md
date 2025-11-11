# Part 1 Tutorial: Understanding Named Entity Extraction

**This tutorial will help you understand the data structure and what you need to do for Part 1.**

## Step 1: Understanding What the Data Looks Like

Let's start by looking at ONE example from the dataset to understand the structure:

```python
from datasets import load_dataset

dataset = load_dataset("hobbes99/fake_movie_reviews_ner_sentiment")
label_list = dataset["train"].features["ner_tags"].feature.names

# Look at the FIRST example
example = dataset["train"][0]
print(example)
```

This will print something like:
```python
{
    'tokens': ['This', 'movie', 'starring', 'Tom', 'Hanks', 'is', 'great'],
    'ner_tags': [0, 0, 0, 1, 2, 0, 0],
    'sentiment': 1
}
```

### What does this mean?

- **tokens**: A list of words in the review (one review broken into individual words)
- **ner_tags**: Numbers that represent the entity type for each word
- **sentiment**: 0 = negative, 1 = positive

### Understanding the `label_list`

When you print `label_list`, you get:
```python
['O', 'B-ACTOR', 'I-ACTOR', 'B-DIRECTOR', 'I-DIRECTOR']
```

This is the "decoder" for the numbers in `ner_tags`. Each number corresponds to a label:
- 0 → 'O' (Outside - not an entity)
- 1 → 'B-ACTOR' (Beginning of an actor's name)
- 2 → 'I-ACTOR' (Inside/continuation of an actor's name)
- 3 → 'B-DIRECTOR' (Beginning of a director's name)
- 4 → 'I-DIRECTOR' (Inside/continuation of a director's name)

### Let's decode the example:

```python
tokens:    ['This', 'movie', 'starring', 'Tom',     'Hanks',   'is', 'great']
ner_tags:  [0,      0,       0,          1,         2,         0,    0]
decoded:   ['O',    'O',     'O',        'B-ACTOR', 'I-ACTOR', 'O',  'O']
```

**This tells us:** "Tom Hanks" is an ACTOR!
- "Tom" starts the actor name (B-ACTOR)
- "Hanks" continues the actor name (I-ACTOR)

## Step 2: What Does "Extract" Mean?

**"Extract" means:** Find all the actors and directors from the review and save them as complete names.

From the example above, we want to **extract**:
- Actor: "Tom Hanks" (combining "Tom" + "Hanks")
- Director: (none in this example)
- Sentiment: 1 (positive)

### Where do we store this information?

We'll create **dictionaries** (or lists) to count how many times each actor/director appears in positive vs negative reviews.

Think of it like this:
```python
actor_counts = {
    "Tom Hanks": {"positive": 5, "negative": 2},
    "Meryl Streep": {"positive": 3, "negative": 1}
}
```

This would mean:
- Tom Hanks appeared in 5 positive reviews and 2 negative reviews
- Meryl Streep appeared in 3 positive reviews and 1 negative review

## Step 3: Understanding "Token Lists to Strings"

### What is a "token list"?

A **token list** is just the list of words in `example['tokens']`.

For the actor "Tom Hanks":
- Token list: `['Tom', 'Hanks']`
- String: `'Tom Hanks'`

### How do we convert token lists to strings?

Use Python's `join()` method:

```python
tokens = ['Tom', 'Hanks']
name = ' '.join(tokens)  # Result: 'Tom Hanks'
```

The `' '.join()` puts a space between each word.

### What about "consecutive I- tags with B- tag"?

This means when you see a B-ACTOR followed by I-ACTOR tags, they belong to the SAME person:

```python
tokens:   ['Tom',     'Hanks',   'and', 'Rita',     'Wilson']
tags:     ['B-ACTOR', 'I-ACTOR', 'O',   'B-ACTOR', 'I-ACTOR']
```

We need to extract TWO actors:
1. "Tom Hanks" (B-ACTOR + I-ACTOR)
2. "Rita Wilson" (B-ACTOR + I-ACTOR)

## Step 4: A Small Working Example

Let's write code to extract entities from ONE example:

```python
from datasets import load_dataset

# Load data
dataset = load_dataset("hobbes99/fake_movie_reviews_ner_sentiment")
label_list = dataset["train"].features["ner_tags"].feature.names

# Get ONE example
example = dataset["train"][0]

# Get the pieces we need
tokens = example['tokens']
ner_tags = example['ner_tags']
sentiment = example['sentiment']

print("Tokens:", tokens)
print("NER tags (numbers):", ner_tags)
print("NER tags (labels):", [label_list[tag] for tag in ner_tags])
print("Sentiment:", "positive" if sentiment == 1 else "negative")
```

## Step 5: Extracting Entities from One Example

Here's code to extract actors and directors from ONE example:

```python
def extract_entities_from_one_example(example, label_list):
    """
    Extract actor and director names from one example.
    Returns a dictionary with 'actors' and 'directors' lists.
    """
    tokens = example['tokens']
    ner_tags = example['ner_tags']

    actors = []
    directors = []

    current_entity = []  # Store tokens for current entity
    current_type = None  # Is it ACTOR or DIRECTOR?

    for i in range(len(tokens)):
        token = tokens[i]
        tag = label_list[ner_tags[i]]  # Convert number to label

        if tag == 'B-ACTOR':
            # Save previous entity if it exists
            if current_entity and current_type == 'ACTOR':
                actors.append(' '.join(current_entity))
            elif current_entity and current_type == 'DIRECTOR':
                directors.append(' '.join(current_entity))

            # Start new actor
            current_entity = [token]
            current_type = 'ACTOR'

        elif tag == 'I-ACTOR':
            # Continue current actor name
            if current_type == 'ACTOR':
                current_entity.append(token)

        elif tag == 'B-DIRECTOR':
            # Save previous entity
            if current_entity and current_type == 'ACTOR':
                actors.append(' '.join(current_entity))
            elif current_entity and current_type == 'DIRECTOR':
                directors.append(' '.join(current_entity))

            # Start new director
            current_entity = [token]
            current_type = 'DIRECTOR'

        elif tag == 'I-DIRECTOR':
            # Continue current director name
            if current_type == 'DIRECTOR':
                current_entity.append(token)

        else:  # tag == 'O' (outside)
            # Save previous entity
            if current_entity and current_type == 'ACTOR':
                actors.append(' '.join(current_entity))
            elif current_entity and current_type == 'DIRECTOR':
                directors.append(' '.join(current_entity))

            # Reset
            current_entity = []
            current_type = None

    # Don't forget the last entity!
    if current_entity and current_type == 'ACTOR':
        actors.append(' '.join(current_entity))
    elif current_entity and current_type == 'DIRECTOR':
        directors.append(' '.join(current_entity))

    return {'actors': actors, 'directors': directors}


# Test it on one example
example = dataset["train"][0]
entities = extract_entities_from_one_example(example, label_list)
print("Extracted actors:", entities['actors'])
print("Extracted directors:", entities['directors'])
print("Sentiment:", "positive" if example['sentiment'] == 1 else "negative")
```

## Step 6: Counting Actors by Sentiment

Now we need to count how many times each actor appears in positive vs negative reviews:

```python
from collections import defaultdict

# Create storage for counts
actor_sentiment = defaultdict(lambda: {"positive": 0, "negative": 0})
director_sentiment = defaultdict(lambda: {"positive": 0, "negative": 0})

# Process FIRST 10 examples (for testing)
for i in range(10):
    example = dataset["train"][i]
    entities = extract_entities_from_one_example(example, label_list)

    # Determine sentiment
    sentiment_label = "positive" if example['sentiment'] == 1 else "negative"

    # Count each actor
    for actor in entities['actors']:
        actor_sentiment[actor][sentiment_label] += 1

    # Count each director
    for director in entities['directors']:
        director_sentiment[director][sentiment_label] += 1

# Show results
print("\nActor counts:")
for actor, counts in actor_sentiment.items():
    print(f"{actor}: {counts['positive']} positive, {counts['negative']} negative")

print("\nDirector counts:")
for director, counts in director_sentiment.items():
    print(f"{director}: {counts['positive']} positive, {counts['negative']} negative")
```

## Step 7: Finding Top 3 Actors/Directors

Once you've counted all entities, you need to find which ones are most associated with positive/negative films:

```python
def calculate_positive_proportion(counts):
    """Calculate what proportion of appearances were in positive reviews."""
    positive = counts['positive']
    negative = counts['negative']
    total = positive + negative

    if total == 0:
        return 0

    return positive / total

# Calculate proportions for each actor
actor_proportions = []
for actor, counts in actor_sentiment.items():
    proportion = calculate_positive_proportion(counts)
    total = counts['positive'] + counts['negative']
    actor_proportions.append({
        'name': actor,
        'positive': counts['positive'],
        'negative': counts['negative'],
        'total': total,
        'positive_proportion': proportion
    })

# Sort by positive proportion (highest first)
actor_proportions.sort(key=lambda x: x['positive_proportion'], reverse=True)

# Show top 3 most positive
print("\nTop 3 actors in POSITIVE films:")
for i in range(min(3, len(actor_proportions))):
    actor = actor_proportions[i]
    print(f"{i+1}. {actor['name']}")
    print(f"   Positive: {actor['positive']}, Negative: {actor['negative']}")
    print(f"   Positive proportion: {actor['positive_proportion']:.2%}\n")

# Sort by positive proportion (lowest first) for negative
actor_proportions.sort(key=lambda x: x['positive_proportion'])

print("\nTop 3 actors in NEGATIVE films:")
for i in range(min(3, len(actor_proportions))):
    actor = actor_proportions[i]
    print(f"{i+1}. {actor['name']}")
    print(f"   Positive: {actor['positive']}, Negative: {actor['negative']}")
    print(f"   Positive proportion: {actor['positive_proportion']:.2%}\n")
```

## Step 8: Putting It All Together

Now you can:

1. **Start small**: Test your code on just 10 examples
2. **Check your output**: Make sure actors/directors are extracted correctly
3. **Scale up**: Change `range(10)` to `range(len(dataset["train"]))` to process all examples
4. **Repeat for directors**: Use the same logic for directors

## Common Pitfalls to Avoid

1. **Don't forget the last entity**: If a review ends with an actor name, you need to save it after the loop
2. **Handle empty entities**: Some reviews might have no actors or directors
3. **Test incrementally**: Don't try to do everything at once. Test with 1 example, then 10, then 100

## What to Reference in the Textbook

You don't need anything specific from the textbook for Part 1! This is about:
- Working with Python lists and dictionaries
- Understanding the BIO tagging scheme (covered in reading questions)
- Counting and sorting

The textbook is more relevant for Parts 2-3 where you'll train models.

## Debugging Tips

If you're stuck, try these:

```python
# Print everything for the first example
example = dataset["train"][0]
print("Tokens:", example['tokens'])
print("Tags:", [label_list[tag] for tag in example['ner_tags']])
print("Sentiment:", example['sentiment'])

# Step through your extraction function with print statements
# to see what's happening at each step
```

## Summary

**What you're doing:**
1. Loop through each review in the training set
2. Find all actor/director names using the BIO tags
3. Record whether that review was positive or negative
4. Count how many positive vs negative reviews each person appeared in
5. Calculate proportions and find the top 3 for each category

**Where information is stored:**
- In dictionaries: `actor_sentiment` and `director_sentiment`
- Each dictionary maps: `name → {"positive": count, "negative": count}`

**What "extract" means:**
- Convert BIO tags back to complete names (e.g., B-ACTOR, I-ACTOR → "Tom Hanks")

You can do this! Start with the small examples above and build up gradually.
