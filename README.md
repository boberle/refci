# REFCI - Regular Expression For Classe Instances

## Introduction

Regular expressions features on arbitrary class instances.  It is somewhat similar to TokensRegex or CQL, but is implemented in Python and offers some specific features.

Here is an example:

Let's say that you have a list of tokens, each token being an object, with
* the form (`form`),
* the part of speech (`pos`),
* the length of the form (`length`):

```python
tokens = [
    Token('The',    'determiner',   3),
    Token('little', 'adjective',    6),
    Token('cats',   'noun',         4),
    Token('eat',    'verb',         3),
    Token('a',      'determiner',   1),
    Token('fish',   'noun',         4),
    Token('.',      'punctuation',  1),
]
```

Then you can search patterns:

- a noun: `[pos="noun"]`
- a noun with more than 3 characters: `[pos="noun" length>3]`
- a noun beginning with a `c`: `[pos="noun" form=/c.*/]`
- a noun with a determiner before it: `[pos="determiner"][pos="noun"]`
- a noun phrase with a determiner, then 0, 1 or more adjectives, then a noun: `[pos="determiner"][pos="adjective"]*[pos="noun"]`
- and much, much more...


## Quickstart

### Setup

Let's define a Token class with a named tuple.  The class has the following attributs:
* `form`,
* `lemma`,
* `pos` (part of speech),
* `is_upper` (whether the form starts with an upper case letter),
* `length`.


```python
from collections import namedtuple

Token = namedtuple('Token', 'form lemma pos is_upper length')

token = Token("cats" , "cat", "noun", False, 4)
print(token.form)
print(token.lemma)
print(token.pos)
print(token.is_upper)
print(token.length)
```

    cats
    cat
    noun
    False
    4


Now let's build some sentences, in the form of a `list` of `Token`s:


```python
tokens = [
    Token('The',    'the',      'determiner',   True,   3),
    Token('little', 'little',   'adjective',    False,  6),
    Token('cats',   'cat',      'noun',         False,  4),
    Token('eat',    'eat',      'verb',         False,  3),
    Token('a',      'a',        'determiner',   False,  1),
    Token('fish',   'fish',     'noun',         False,  4),
    Token('.',      '.',        'punctuation',  False,  1),
    Token('They',   'they',     'pronoun',      True,   4),
    Token('are',    'be',       'verb',         False,  3),
    Token('happy',  'happy',    'adjective',    False,  5),
    Token(':',      ':',        'punctuation',  False,  1),
    Token('they',   'they',     'pronoun',      False,  4),
    Token('like',   'like',     'verb',         False,  4),
    Token('this',   'this',     'determiner',   False,  4),
    Token('Meal',   'meal',     'noun',         True,  4),
    Token('.',      '.',        'punctuation',  False,  1),
    Token('.',      '.',        'punctuation',  False,  1),
    Token('.',      '.',        'punctuation',  False,  1),
]
```

Let's import `refci` `Pattern` class:


```python
from refci import Pattern
```

And now we can start search for patterns.  To build a pattern, just use:

```python
pat = Pattern('[pos="determiner"][pos="noun"]')
```

There are 4 main functions you can use:
* `pat.search(tokens)`: find the first occurrence of the pattern in the tokens,
* `pat.match(tokens)`: the pattern must be at the beginning of the tokens,
* `pat.fullmatch(tokens)`: the pattern must match the whole set of tokens
* `pat.finditer(tokens)`: loop over all the patterns that match in the tokens (by default not overlapping).

### Simple patterns

So, two find all the **determiners followed by a noun**:


```python
pat = Pattern('[pos="determiner"][pos="noun"]')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['a', 'fish']
    ['this', 'Meal']


Note here that `seq` is a `list` of tokens. You can get **position indices** if you prefer:


```python
pat = Pattern('[pos="determiner"][pos="noun"]')
for seq in pat.finditer(tokens, return_objects=False):
    print(seq)
```

    (4, 6)
    (13, 15)


If the determiner must have **less than 4 characters**, just add a condition:


```python
pat = Pattern('[pos="determiner" length<4][pos="noun"]')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['a', 'fish']


If the **noun must be capitalized**:


```python
pat = Pattern('[pos="determiner"][pos="noun" is_upper=True]')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['this', 'Meal']


If the noun must have a specific lemma, **determined with a regular expression**:


```python
pat = Pattern('[pos="determiner"][]*?[pos="noun" lemma=/cats?/]')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['The', 'little', 'cats']


Now we want noun phrase with a determiner and a noun, and **0 or 1 adjective in the middle**:


```python
pat = Pattern('[pos="determiner"][pos="adjective"]?[pos="noun"]')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['The', 'little', 'cats']
    ['a', 'fish']
    ['this', 'Meal']


Or, really, **any word in the middle**:


```python
pat = Pattern('[pos="determiner"][]*?[pos="noun"]')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['The', 'little', 'cats']
    ['a', 'fish']
    ['this', 'Meal']


### Variables

You can define **variables**.  For example, if you want to search for contiguous words of the same length (even if overlapping):


```python
pat = Pattern('[variable<-length][length==$variable]')
for seq in pat.finditer(tokens, overlapping=True):
    print([token.form for token in seq])
```

    ['they', 'like']
    ['like', 'this']
    ['this', 'Meal']
    ['.', '.']
    ['.', '.']


Or sequence of 2 words in which the second word is longer than the first:


```python
pat = Pattern('[variable<-length][length>$variable]')
for seq in pat.finditer(tokens, overlapping=True):
    print([token.form for token in seq])
```

    ['The', 'little']
    ['a', 'fish']
    ['.', 'They']
    ['are', 'happy']
    [':', 'they']


### Groups

You can define groups, either to offer **an alternative** (**OR operator**), for example if you want either a full noun phrase or a pronoun:


```python
pat = Pattern('( [pos="determiner"][]*?[pos="noun"] | [pos="pronoun"] )')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['The', 'little', 'cats']
    ['a', 'fish']
    ['They']
    ['they']
    ['this', 'Meal']


or to **capture only parts of the pattern**, for example if your only interested in the noun, not the determiner or the adjectives:


```python
pat = Pattern('[pos="determiner"][]*?(?P<interesting>[pos="noun"])')
for _ in pat.finditer(tokens):
    group_indices = pat.get_group('interesting')
    print(group_indices)
    group_tokens = pat.get_group('interesting', objs=tokens)
    print([token.form for token in group_tokens])
```

    (2, 3)
    ['cats']
    (5, 6)
    ['fish']
    (14, 15)
    ['Meal']


### Quantifiers

You can use the quantifiers familiar to any regular expression engine. For example, with **no quantifier** after the ponctuation:


```python
pat = Pattern('[pos="noun"][pos="punctuation"]')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['fish', '.']
    ['Meal', '.']


with a `*` (0, 1 or more punctuation):


```python
pat = Pattern('[pos="noun"][pos="punctuation"]*')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['cats']
    ['fish', '.']
    ['Meal', '.', '.', '.']


with a `?` (0 or 1 punctuation):


```python
pat = Pattern('[pos="noun"][pos="punctuation"]?')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['cats']
    ['fish', '.']
    ['Meal', '.']


with a `+` (1 or more punctuations):


```python
pat = Pattern('[pos="noun"][pos="punctuation"]+')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])

```

    ['fish', '.']
    ['Meal', '.', '.', '.']


with a custom number of punctuation (here between 2 and 3):


```python
pat = Pattern('[pos="noun"][pos="punctuation"]{2,3}')
for seq in pat.finditer(tokens):
    print([token.form for token in seq])
```

    ['Meal', '.', '.', '.']


### `finditer` vs `search` vs `[full]match`

Rather than `finditer`, you can use `search` to get the first occurrence:


```python
pat = Pattern('[pos="noun"]')
seq = pat.search(tokens)
print([token.form for token in seq])
```

    ['cats']


or the first occurrence after a certain point:


```python
pat = Pattern('[pos="noun"]')
seq = pat.search(tokens, start=10)
print([token.form for token in seq])
```

    ['Meal']


The `match` function will only match at the beginning of the tokens:


```python
pat = Pattern('[pos="determiner"][pos="adjective"]')
seq = pat.match(tokens)
print([token.form for token in seq])
```

    ['The', 'little']


While the `fullmatch` will only match for the whole token sequence:


```python
pat = Pattern('[pos="determiner"][pos="adjective"]')
seq = pat.fullmatch(tokens)
print(seq)
```

    None


## More explanations


See the [`guide.html`](https://htmlpreview.github.io/?https://github.com/boberle/refci/blob/master/guide.html) file for more examples!

See the [`guide.ipynb`](guide.ipynb) file to start to play with.


## Licence

Distributed under the terms of the Mozilla Public License 2.0, see the `LICENSE` file.

