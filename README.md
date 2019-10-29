# REFCI - Regular Expression For Classe Instances

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

See the **quick start** tour in the [`guide.html`](guide.html) for more examples!

To use it, just download the `refci.py` module and import the `Pattern` class.  Please see the **quick start** tour in the [`guide.html`](guide.html) for more information, and the [`guide.ipynb`](guide.ipynb) to start to play with.


Distributed under the terms of the Mozilla Public License 2.0, see the `LICENSE` file.

