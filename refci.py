"""
REFCI - Regular Expression For Classe Instances

Regular expressions features on arbitrary class instances.  It is somewhat
similar to TokensRegex or CQL, but is implemented in Python and offers some
specific features.

Here is an example:

Let's say that you have a list of tokens, each token being an object, with
* the form (`form`),
* the part of speech (`pos`),
* the length of the form (`length`):

    tokens = [
        Token('The',    'determiner',   3),
        Token('little', 'adjective',    6),
        Token('cats',   'noun',         4),
        Token('eat',    'verb',         3),
        Token('a',      'determiner',   1),
        Token('fish',   'noun',         4),
        Token('.',      'punctuation',  1),
    ]

Then you can search patterns:
- a noun: `[pos="noun"]`
- a noun with more than 3 characters: `[pos="noun" length>3]`
- a noun beginning with a `c`: ``[pos="noun" form=/c.*/]`
- a noun with a determiner before it: `[pos="determiner"][pos="noun"]`
- a noun phrase with a determiner, then 0, 1 or more adjectives, then a noun:
  `[pos="determiner"][pos="adjective"]*[pos="noun"]`
- and much, much more...

Please see the `user_guide.html` for more information, and the
`user_guide.ipynb` to start to play with.
"""

__version__ = '1.0.0'

# Copyright 2019 Bruno Oberle
# Mozilla Public License 2.0, see the LICENSE file

import re
import collections
import sys

import pandas as pd

def _get_objs(objs, start, stop=None):
    """Return an obj/Series of objs if `stop` is None, or a slice/Dataframe.

    This works for list and dataframes.
    """
    if isinstance(objs, pd.core.frame.DataFrame):
        if stop is None:
            return objs.iloc[start]
        return objs.iloc[start:stop]
    else:
        if stop is None:
            return objs[start]
        return objs[start:stop]


class Pattern:
    """Represents a compiled parttern to be matched against a list of objects.

    Attributes
    ----------
    string: str
        The pattern string.  It is compiled at init time.  Read only.
    return_objects: bool (True)
        If True, the match and search functions return objects instance of the
        tuple (start, stop).
    """

    def __init__(self, string, return_objects=True):
        self._string = string
        self.return_objects = return_objects
        self._atom = _compile(string) # the root Atom or GroupAtom
        if isinstance(self._atom, Atom):
            self._atom = GroupAtom([self._atom])
        self._atom.index = 0
        self._vars = dict() # variables used in the pattern
        self._group_atoms = dict() # capture groups, keys are both indices
            # (int, for all groups) and names (for named groups)
        self._finalize()
        self._set_last_atom()

    @property
    def string(self):
        """Return the string used to compile the pattern."""
        return self._string

    def set_group_mode(self, mode):
        """Set the group mode, either: 'normal' or 'group'.  See documentation.
        
        Apply to the root atom, which will distribute the mode to its children,
        if any.
        """
        self._atom.set_mode(mode)

    def _finalize(self, atom=None):
        """Finalize the compilation by distributing the variable dic `_vars`
        and filling the group dic `_group_atoms`.
        """
        if atom is None:
            atom = self._atom
        if isinstance(atom, Atom):
            atom.add_vars(self._vars)
        elif isinstance(atom, GroupAtom):
            if atom.index is not None:
                self._group_atoms[atom.index] = atom
            if atom.name is not None:
                self._group_atoms[atom.name] = atom
            for subatom in atom.subatoms:
                self._finalize(subatom)

    def _set_last_atom(self):
        """Set `is_last` to True to the last atom.
        
        This property is used by `fullmatch()`."""
        def get_last_atom(atom):
            if isinstance(atom, Atom):
                return atom
            else:
                atom = atom.subatoms[-1]
                return get_last_atom(atom)
        get_last_atom(self._atom).is_last = True

    def get_group(self, key, objs=None):
        group = self._group_atoms[key]
        if objs:
            return _get_objs(objs, group.start, group.stop)
        else:
            return group.start, group.stop

    def _get_match(self, **kwargs):
        """Try to match the pattern against objects and return the stop index,
        or `None` if nothing is found.

        The arguments are directly passed to `self._atom.match`.
        """
        try:
            return next(self._atom.match(**kwargs))
        except StopIteration:
            return None

    def match(self, objs, start=0, return_objects=None, full=False):
        """Try to match the pattern against `objs` at index `start`.
        
        Return `(start, stop)`, or `None` if the pattern doesn't match.

        Parameters
        ----------
        objs: a list or a tuple of class instances (not a generator)
            The objects to match the pattern against.
        start: int (0)
            The match must start at this index.
        return_objects: bool (None)
            If True, return a list of objects (or None if no match).  If False
            return a tuple `(start, stop)` (`stop` is not in the match).  If
            `None`, use the value of the instance attribute of the same name.
        full: bool (False)
            The pattern must match at the `start` **and** at the end of `objs`.
        """
        j = self._get_match(objs=objs, i=start, full=full)
        if j is None:
            return None
        if return_objects or (return_objects is None and self.return_objects):
            return _get_objs(objs, start, j)
        return start, j

    def search(self, objs, start=0, return_objects=None):
        """Like `match()`, but returns the first sequence of objects found, it
        doesn't need to start exactly at `start`.
        """
        for i in range(start, len(objs)):
            j = self._get_match(objs=objs, i=i)
            if j is not None:
                if return_objects \
                        or (return_objects is None and self.return_objects):
                    return _get_objs(objs, i, j)
                return i, j
        return None

    def fullmatch(self, objs, start=0, return_objects=None):
        """Short hand for `match()` with the `full` parameter set to `True`."""
        return self.match(objs, start, return_objects, full=True)

    def finditer(self, objs, start=0, return_objects=None, overlapping=False):
        """Repetitively call `search()` until no sequence is found.

        Set `overlapping` to True (default is False) to get overlapping
        sequences.
        """
        start = start
        while start < len(objs):
            pos = self.search(objs, start=start, return_objects=False)
            if pos is None:
                break
            start, stop = pos
            if return_objects \
                    or (return_objects is None and self.return_objects):
                yield _get_objs(objs, start, stop)
            else:
                yield start, stop
            if overlapping:
                start += 1
            else:
                start = stop

    def get_string(self):
        return self._atom.get_string()

class Spec:
    """Represent a specification of an atom.

    For example, in `[foo="bar" baz>5]`, both `foo="bar"` and `baz>5` are
    specifications.

    Attributes
    ----------
    kind: str, one of num regex string bool varsetting eval
        The type of specification.
    attr: str
        The attribute name of the object (`foo` and `baz` in the example
        above).
    operator: str
        The value depends of the kind:
        * num|eval: '==', '!=', '>', '<', '>=', '<='
        * varsetting: ''
        * otherwise: '!=', '='
    value: misc
        The value against which the attribute is tested ("bar" and 5 in the
        example above).
    """
    def __init__(self, kind, attr, operator, value):
        self.kind = kind  # num regex string bool
        assert self.kind in ('num', 'regex', 'string', 'bool', 'varsetting',
            'eval', 'subpattern'), self.kind
        self.attr = attr
        self.value = value
        if self.kind == 'regex':
            self.value = re.compile(self.value)
        elif self.kind == 'bool':
            self.value = bool(self.value)
        self.operator = operator
        if self.kind in ('num', 'eval'):
            assert self.operator in ('==', '!=', '>', '<', '>=', '<='), \
                self.operator
        elif self.kind in ('varsetting',):
            assert self.operator in ('',), self.operator
        elif self.kind in ('subpattern',):
            assert self.operator in ('=', '==', '~', '!=', '!==', '!~'), \
                self.operator
        else:
            assert self.operator in ('!=', '='), self.operator

    def test(self, obj):
        """Test the value against the object, according to the specifications
        given at init time.  Return True or False.
        """
        if isinstance(obj, pd.core.series.Series):
            val = obj[self.attr]
        else:
            val = getattr(obj, self.attr)
        if self.kind == 'num':
            if self.operator == '==':   return val == self.value
            elif self.operator == '!=': return val != self.value
            elif self.operator == '>=': return val >= self.value
            elif self.operator == '<=': return val <= self.value
            elif self.operator == '>':  return val >  self.value
            elif self.operator == '<':  return val <  self.value
        elif self.kind == 'eval':
            var_value = self._vars[self.value]
            if self.operator == '==':   return val == var_value
            elif self.operator == '!=': return val != var_value
            elif self.operator == '>=': return val >= var_value
            elif self.operator == '<=': return val <= var_value
            elif self.operator == '>':  return val >  var_value
            elif self.operator == '<':  return val <  var_value
        elif self.kind == 'varsetting':
            self._vars[self.value] = val
            return True
        elif self.kind == 'regex':
            res = bool(self.value.fullmatch(val))
        elif self.kind == 'string':
            res = val == self.value
        elif self.kind == 'bool':
            res = bool(val) == self.value
        elif self.kind == 'subpattern':
            neg = self.operator.startswith('!')
            op = self.operator[1:] if neg else self.operator
            func = {
                '=': Pattern.match,
                '==': Pattern.fullmatch,
                '~': Pattern.search,
            }[op]
            match = func(self.value, objs=val)
            return match == None if neg else match is not None
        return not res if self.operator == '!=' else res

    def get_string(self):
        """Pretty string that represents the specificiation."""
        if self.kind == 'num':
            return '%s%s%s' % (self.attr, self.operator, str(self.value))
        elif self.kind == 'eval':
            return '%s%s$%s' % (self.attr, self.operator, self.value)
        elif self.kind == 'varsetting':
            return '%s<-%s' % (self.value, self.attr)
        elif self.kind == 'regex':
            return '%s%s/%s/' % (self.attr, self.operator, self.value.pattern)
        elif self.kind == 'string':
            return '%s%s"%s"' % (self.attr, self.operator, self.value)
        elif self.kind == 'bool':
            return '%s%s%s' % (self.attr, self.operator, str(self.value))
        elif self.kind == 'subpattern':
            return '%s%s{%s}' % (self.attr, self.operator,
                str(self.value.get_string()))
        assert False

"""Mapping of string converter to a tuple `(lower, upper, hungriness)`.  Used
by the init function of the Atom class.
"""
_QUANTIFIERS = {
    ''  : (1, 1          , 'lazy'),
    '*' : (0, sys.maxsize, 'greedy'),
    '?' : (0, 1,           'greedy'),
    '+' : (1, sys.maxsize, 'greedy'),
    '*?': (0, sys.maxsize, 'lazy'),
    '??': (0, 1,           'lazy'),
    '+?': (1, sys.maxsize, 'lazy'),
    '*+': (0, sys.maxsize, 'possessive'),
    '?+': (0, 1,           'possessive'),
    '++': (1, sys.maxsize, 'possessive'),
}

class BaseAtom:
    """I'm the base class for both Atom and GroupAtom.
    
    An atom is a list of specifications that apply to one token.  In
    `[foo="bar"][baz>5]`, there are two atoms (between square brackets).

    A GroupAtom a is group of several atoms.  In the pattern string, they may
    be invisible, but any sequence of two or more atoms is enclosed in a group.

    GroupAtom may be captured `([][]|[])`, captured and named
    `(?P<name>[][]|[])` or not captured `(?:[][]|[])`.

    Attributes
    ----------
    lower: int
        The minimum number of repetitions.
    upper: int
        The maximum number of repetitions.
    hungriness: str, one of 'lazy', 'greedy', 'possessive'
        How the atom or group feels about eating objects.
        Default is the tuple given by the '' key in _QUANTIFIERS.
    """
    def __init__(self):
        self.lower, self.upper, self.hungriness = _QUANTIFIERS['']

    def match(self, objs, i, full=False):
        """Try to match the atom or group against `objs` (a list of class
        instances) starting at `i`.

        Yield all possible `stop` indices, in order of preference (from left to
        right if lazy, from right to left if greedy, the last possible index if
        possessive.

        If `full` (default False), the atom (or the last atom of the group)
        must match the end of `objs`.
        """
        if self.hungriness == 'greedy':
            iterator = self._go_greedy
        elif self.hungriness == 'possessive':
            iterator = self._go_possessive
        elif self.hungriness == 'lazy':
            iterator = self._go_lazy
        else:
            assert False, self.hungriness
        yield from iterator(objs, i, full=full)

    def set_quantifier(self, quantifier):
        """Set the lower, upper and hungriness instance attributes by looking
        the `quantifier` argument in the `_QUANTIFIERS` dictionary.  The
        argument may also be a string of the form `/{(\d*),(\d*)}([?+]?)/`.
        """
        m = re.fullmatch(r'{(\d*),(\d*)}([?+]?)', quantifier)
        if m:
            lower, upper, hungriness = m.group(1,2,3)
            if not lower:
                lower = 0
            if not upper:
                upper = sys.maxsize
            lower = int(lower)
            upper = int(upper)
            hungriness = \
                {'': 'greedy', '+': 'possessive', '?': 'lazy'}[hungriness]
        else:
            lower, upper, hungriness = _QUANTIFIERS[quantifier]
        self.lower, self.upper, self.hungriness = lower, upper, hungriness


    def quant2string(self):
        """Pretty string representing the quantification."""
        if (self.lower, self.upper) == (1, 1):
            return ""
        if (self.lower, self.upper) == (0, 1):
            res = "?"
        elif (self.lower, self.upper) == (0, sys.maxsize):
            res = "*"
        elif (self.lower, self.upper) == (1, sys.maxsize):
            res = "+"
        else:
            res = "{%s,%s}" % (
                str(self.lower),
                "" if self.upper == sys.maxsize else str(self.upper))
        if self.hungriness == "lazy":
            res += "?"
        elif self.hungriness == "greedy":
            res += ""
        elif self.hungriness == "possessive":
            res += "+"
        return res


class Atom(BaseAtom):
    """See BaseAtom."""

    def __init__(self):
        super().__init__()
        self._specs = [] # the Specification's
        self.is_last = False # last atom in pattern?


    def _match(self, objs, i, full=False):
        """Test the atom against `objs[i]` and return the `stop` index, or
        `None` if there is no match.

        If `full` is True (default False) and the atom `is_last`, then match
        only at the end of `objs`.
        """
        for spec in self._specs:
            if not spec.test(_get_objs(objs, i)):
                return None
            if full:
                if self.is_last and not i == len(objs) - 1:
                    return None
        return i + 1

    def _go_lazy(self, objs, i, full=False):
        """Yield all possible `stop` indices that match the atom, from left to
        right.
        """
        if self.lower == 0:
            yield i
        for count, x in enumerate(range(self.upper), start=1):
            if not count >= self.lower: # PATCH
                continue                # PATCH
            if not i + x < len(objs):
                break
            if self._match(objs, i+x, full=full) is not None:
                yield i + x + 1
            else:
                break

    def _go_greedy(self, objs, i, full=False):
        """Yield all possible `stop` indices that match the atom, from right to
        left.
        """
        for x in reversed(list(self._go_lazy(objs, i, full))):
            yield x

    def _go_possessive(self, objs, i, full=False):
        """Yield the last possible `stop` index that matches the atom.  Yield
        at most one value."""
        l = list(self._go_lazy(objs, i, full=full))
        if l:
            yield l[-1]

    def add_vars(self, vars_):
        """Add the variable dictionary to all the specification.

        This dictionary is the variable dictionary of the pattern (`_vars`).
        """
        for spec in self._specs:
            spec._vars = vars_

    def add_spec(self, kind, attr, op, val):
        """Add a specification.  The parameters are those of the Spec class."""
        self._specs.append(Spec(kind, attr, op, val))

    def get_string(self):
        """Pretty string represting the atom and its specs."""
        return '[%s]%s' % (
            " ".join(s.get_string() for s in self._specs),
            self.quant2string()
        )


class GroupAtom(BaseAtom):
    """See BaseAtom.  Represent a group of atoms.

    Attributes
    ----------
    subatoms: list of Atom's (None)
        Atoms forming the group.
    index: int (None)
        The group index, if any.
    name: str (None)
        The group name, if any.
    operator: str, one of 'and', 'or' ('and')
        The relation between the atoms.
    start: None
        The start index of the group.
    stop: None
        The stop index of the group.

    Notes
    -----
    The default mode is 'normal'.
    """

    def __init__(self, subatoms=None, index=None, name=None):
        super().__init__()
        self.subatoms = subatoms
        self.operator = 'and'
        self.name = name
        self.index = index
        self.start = None
        self.stop = None
        self.set_mode('normal')

    def set_mode(self, mode):
        """Set the mode.  See the documentation.
        
        Implementation
        --------------
        Set the three attributes `_go_lazy/greedy/possessive` to
        `_go_lazy/..._<mode>`.
        """
        assert mode in ('normal', 'group')
        code = mode.upper()
        self._go_lazy = getattr(self, '_go_lazy__%s' % code)
        self._go_greedy = getattr(self, '_go_greedy__%s' % code)
        self._go_possessive = getattr(self, '_go_possessive__%s' % code)
        for subatom in self.subatoms:
            if isinstance(subatom, GroupAtom):
                subatom.set_mode(mode)

    def _match(self, objs, i, full=False):
        """Yield all possible `stop` indices that match the group against the
        objects, starting at `i`, according to the operator (and, or)."""
        self.start = i
        if self.operator == 'and':
            #yield from self._and_match(objs, i, full=full)
            for x in self._and_match(objs, i, full=full):
                self.stop = x
                yield x
        elif self.operator == 'or':
            #yield from self._or_match(objs, i, full=full)
            for x in  self._or_match(objs, i, full=full):
                self.stop = x
                yield x
        else:
            assert False, self.operator

    def _or_match(self, objs, i, full=False):
        """See `_match()`."""
        for atom in self.subatoms:
            yield from atom.match(objs, i, full=full)

    def _and_match(self, objs, i, a=0, full=False):
        """See `_match()`."""
        if a < len(self.subatoms):
            for j in self.subatoms[a].match(objs, i, full=full):
                yield from self._and_match(objs, j, a+1, full=full)
        else:
            yield i

    def _go_lazy__NORMAL(self, objs, i, full=False, _recur_count=0):
        """Yield all possible `stop` indices that match the atom, from left to
        right.

        Implementation
        --------------
        Take the first (preferred) `stop` index for the group, and go recursive
        (for repetition 2, if any) and take the first preferred `stop` index
        for the group (repetition 2), and so on.

        If nothing match, backtrack and try the second preferred `stop` index
        for the group, etc.

        Note that there is **no** trying to maximize group repetition.
        """
        if _recur_count > self.upper:
            return
        if _recur_count >= self.lower:
            yield i
        for y in self._match(objs, i, full=full):
            found = False
            for x in self._go_lazy(
                    objs, y, full=full, _recur_count=_recur_count+1):
                found = True
                yield x
            if found:
                break

    def _go_greedy__NORMAL(self, objs, i, full=False):
        """Yield all possible `stop` indices that match the atom, from right to
        left.

        Implementation
        --------------
        Reverse the returned indices of `_go_lazy`.
        """
        for x in reversed(list(self._go_lazy(objs, i, full))):
            yield x

    def _go_possessive__NORMAL(self, objs, i, full=False):
        """Yield the last possible `stop` index that matches the atom.  Yield
        at most one value.

        Implementation
        --------------
        Yield the the last value returned by `_go_lazy`, if any.
        """
        x = None
        for x in list(self._go_lazy(objs, i, full=full)):
            pass
        if x is not None:
            yield x
        # other way to do that:
        #l = list(self._go_lazy(objs, i, full=full))
        #if l:
        #    yield l[-1]


    def _go_lazy__GROUP(self, objs, i, full=False, _recur_count=0, _raw=False):
        """Yield all possible `stop` indices that match the atom, from left to
        right.

        Implementation
        --------------
        Like the NORMAL version, but the algorithm try to maximize group
        repetition.
        """
        if _recur_count > self.upper:
            return
        if _recur_count >= self.lower:
            yield (_recur_count, i) if _raw else i
        for y in self._match(objs, i, full=full):
            for x in self._go_lazy(objs, y, full=full,
                    _recur_count=_recur_count+1, _raw=_raw):
                yield x

    def _go_greedy__GROUP(self, objs, i, full=False):
        """Yield all possible `stop` indices that match the atom, from right to
        left.

        Implementation
        --------------
        Reverse the returned indices of `_go_lazy`.
        """
        l = list(self._go_lazy(objs, i, full, _raw=True))
        #print(sorted(l, reverse=True, key=lambda x: x[0]))
        for count, stop in sorted(l, reverse=True, key=lambda x: x[0]):
            yield stop

    def _go_possessive__GROUP(self, objs, i, full=False):
        """Yield the last possible `stop` index that matches the atom.  Yield
        at most one value.

        Implementation
        --------------
        Yield the first value returned by `_go_greedy`, if any.
        """
        try:
            yield next(self._go_greedy(objs, i, full))
        except StopIteration:
            pass

    def get_string(self):
        """Pretty string represting the group and its atoms."""
        return '(%s%s)%s' % (
            ("?P<%s> " % self.name) if self.name else "",
            (" " if self.operator == 'and' else ' | ')
                .join(a.get_string() for a in self.subatoms),
            self.quant2string()
        )

##### parser ###########################################################

"""Contain the cached pattern strings."""
_cache = dict()

def _compile(string, cache=True):
    """Compile the pattern.  If `string` is in the cache, the pattern is
    returned and reused, unless `cache` is False.

    Return a Pattern object.
    """
    global _cache
    if cache and string in _cache:
        return _cache[string]
    pattern = _parse(string=string)
    if cache:
        _cache[string] = pattern
    return pattern

"""Objet that represent a token of the pattern string."""
Token = collections.namedtuple('Token', ['kind', 'feat'])

"""Regex to parse the pattern string"""
_token_regex = None

def _create_token_regex():
    """Create the `_token_regex` variable."""
    res = []
    #def escape(string, chars='?*+'):
    #    for char in chars:
    #        string = string.replace(char, '\\'+char)
    #    return string
    quantifiers = '(?:(?:[?*+]|\{\d*,\d*\})[?+]?)?'
    base_op = '=|!='
    all_op = '==|!=|<|>|<=|>='
    subpat_op = '!?(?:~|==|=)'
    specs = [
        ('SPACE', r'\s+'),
        ('OPEN_NON_CAPTURING_GROUP',  r'\(\?:'),
        ('OPEN_CAPTURING_GROUP',  r'\((?:\?P<(?P<group_name>\w+)>)?'),
        ('CLOSE_GROUP', r'\)\s*(?P<group_quant>%s)' % quantifiers),
        ('OPEN_ATOM',  r'\['),
        ('CLOSE_ATOM', r'\]\s*(?P<atom_quant>%s)' % quantifiers),
        ('SPEC_STR',
            r'(?P<str_attr>\w+)(?P<str_op>%s)"(?P<str_val>[^"]*)"' % base_op),
        ('SPEC_REGEX',
            r'(?P<regex_attr>\w+)(?P<regex_op>%s)/(?P<regex_val>[^"]*)/' % base_op),
        ('SPEC_NUM',
            r'(?P<num_attr>\w+)(?P<num_op>%s)(?P<num_val>[0-9]+(?:\.[0-9]+)?)' % all_op),
        ('SPEC_SET',
            r'(?P<set_varname>\w+)<-(?P<set_attr>\w+)'),
        ('SPEC_EVAL',
            r'(?P<eval_attr>\w+)(?P<eval_op>%s)\$(?P<eval_varname>\w+)' % all_op),
        ('SPEC_BOOL',
            r'(?P<bool_attr>\w+)(?P<bool_op>%s)(?P<bool_val>True|T|t|true|False|F|f|false)' % base_op),
        ('SPEC_SUB',
            r'(?P<sub_attr>\w+)(?P<sub_op>%s)\{(?P<sub_val>[^}]+)\}' % subpat_op),
        ('OR',r'\|'),
        ('MISMATCH',r'.'),
    ]
    global _token_regex
    _token_regex = re.compile('|'.join('(?P<%s>%s)' % pair for pair in specs))

_create_token_regex()

"""Dictionary of the form 'bool_op' -> 'op'.  Take all group names in
`_token_regex` and split them at the underscore.

This is used to simplify the way Spec() is called: argument names are what is
after the underscore (`op` for operator in the example.

This is necessary because two group can't have the same name in a Python
regular expression.
"""
_regex_groups = {
    name:name[name.index('_')+1:]
    for name in _token_regex.groupindex if '_' in name and name.islower()
}
#print(_regex_groups)

def _tokenize(string):
    """Yield a Token."""
    capturing_group_count = 1
    for m in _token_regex.finditer(string):
        kind = m.lastgroup
        if kind == 'SPACE':
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError("parsing error: '%s' unexpected in '%s'" %
                (m.group(kind), string))
        feat = {
            pretty_name:m.group(name)
            for name, pretty_name in _regex_groups.items()
            if m.group(name) is not None
        }
        token = Token(kind, feat)
        if kind == 'SPEC_BOOL':
            token.feat['val'] = token.feat['val'][0] in 'Tt'
        if kind == 'OPEN_CAPTURING_GROUP':
            token.feat['index'] = capturing_group_count
            capturing_group_count += 1
            if 'name' not in token.feat:
                token.feat['name'] = None
        yield token

def _parse(tokenizer=None, string=None):
    """Parse the pattern string.

    The function is recursive and returns either an Atom or a GroupAtom.
    GroupAtom's don't necessarily appear in the string, they are formed when
    two or more Atom are consecutive.

    When first called, pass the `string` argument.  When going recursive, the
    function will pass to itself a tokenizer, which is the `_tokenize()`
    function with the iterator cursor at the next token to be read.
    """
    assert tokenizer or string
    if tokenizer is None:
        tokenizer = iter(_tokenize(string))
    operator = 'and'
    pending = []
    elements = []
    force_group = False
    atom = None
    while True:
        try:
            kind, feat = next(tokenizer)
        except StopIteration:
            break
        if kind == 'OPEN_CAPTURING_GROUP':
            group = _parse(tokenizer)
            group.name = feat['name']
            group.index = feat['index']
            pending.append(group)
        elif kind == 'OPEN_NON_CATPURING_GROUP':
            group = _parse(tokenizer)
            pending.append(group)
        elif kind == 'CLOSE_GROUP':
            force_group = True
            break
        elif kind == 'OPEN_ATOM':
            atom = Atom()
        elif kind == 'CLOSE_ATOM':
            assert atom
            atom.set_quantifier(feat['quant'])
            pending.append(atom)
            atom = None
        elif kind == 'OR':
            operator = 'or'
            assert pending
            if len(pending) > 1:
                elements.append(GroupAtom(pending))
            else:
                elements.append(pending[0])
            pending = []
        elif kind == 'SPEC_STR':
            assert atom
            atom.add_spec('string', feat['attr'], feat['op'], feat['val'])
        elif kind == 'SPEC_REGEX':
            assert atom
            atom.add_spec('regex', feat['attr'], feat['op'], feat['val'])
        elif kind == 'SPEC_BOOL':
            assert atom
            atom.add_spec('bool', feat['attr'], feat['op'], feat['val'])
        elif kind == 'SPEC_NUM':
            assert atom
            val = feat['val']
            val = float(val) if '.' in val else int(val)
            atom.add_spec('num', feat['attr'], feat['op'], val)
        elif kind == 'SPEC_SET':
            atom.add_spec('varsetting', feat['attr'], '', feat['varname'])
        elif kind == 'SPEC_EVAL':
            atom.add_spec('eval', feat['attr'], feat['op'], feat['varname'])
        elif kind == 'SPEC_SUB':
            subpat = Pattern(feat['val'])
            atom.add_spec('subpattern', feat['attr'], feat['op'], subpat)
    if operator == 'and':
        if pending:
            elements.extend(pending)
        assert elements
    else:  # or
        if len(pending) > 1:
            elements.append(GroupAtom(pending))
        elif len(pending) == 1:
            elements.append(pending[0])
        assert len(elements) > 1
        force_group = True
    if force_group:
        group = GroupAtom(elements)
        group.operator = operator
        group.set_quantifier(feat['quant'])
        return group
    if len(elements) > 1:
        return GroupAtom(elements)
    else:
        return elements[0]


def make_test_pattern(pattern):
    # note: don't replace number because otherwise you can't use {1,2}
    pattern = re.sub(r'([a-zA-Z])', r'[data="\1"]', pattern)
    pattern = pattern.replace('.', '[]')
    pat = Pattern(pattern)
    return pat

def make_test_data(string, pattern=None):
    """Build a test object list with a test pattern.

    I'm a helper function to help you to test and play with the REFCI module.

    Return a tuple `(objs, pattern)`.  If `pattern` is None, then it is not
    build, and the returned pattern is None.

    The `string` argument is string of lower and uppercase letters: `aaBdefF`.
    Each letter gives birth to an object with two attribute: `data` containing
    the lowercase version of the letter, and `upper` indicating if the original
    letter is uppercase or not.

    The pattern is the pattern to match, using only letters: `(a+b|f.)`.  This
    will build the pattern as `([data="a"]+[data="b"]|[data="f"][])`.

    If you want a more complex pattern, you must define it yourself.
    """
    class Obj:
        def __init__(self, data):
            self.data = data.lower()
            self.upper = data.isupper()
        def __str__(self):
            if self.upper:
                return self.data.upper()
            return self.data

    objs = [Obj(x) for x in list(string)]
    pat = make_test_pattern(pattern) if pattern else None
    return (objs, pat) if pat is not None else objs


def build_highlight_series(objs, pat, overlapping=False, groupby=None):
    hl = pd.Series(False, index=objs.index)
    if groupby:
        for name, data in objs.groupby(groupby):
            for res in pat.finditer(data, overlapping=overlapping):
                hl = hl | pd.Series(True, index=res.index)
    else:
        for res in pat.finditer(objs, overlapping=overlapping):
            hl = hl | pd.Series(True, index=res.index)
    return hl

def build_bio_series(objs, pat, groupby=None):
    bio = pd.Series("O", index=objs.index)
    if groupby:
        for name, data in objs.groupby(groupby):
            for res in pat.finditer(data, overlapping=False):
                bio.update(pd.Series(
                    ["B"] + ["I"] * (len(res)-1), index=res.index))
    else:
        for res in pat.finditer(objs, overlapping=False):
            bio.update(pd.Series(
                ["B"] + ["I"] * (len(res)-1), index=res.index))
    return bio

def by_group(objs, pat, groupby, func='finditer'):
    if func == 'finditer':
        for group, data in objs.groupby(groupby):
            yield from pat.finditer(data)
    else:
        func = getattr(pat, func)
        for group, data in objs.groupby(groupby):
            res = func(data)
            if res is not None:
                yield res

def main():

    if False: # OK
        objs, pat = make_test_data(
            'aaAbabaabcDeeF',
            '(a++b??)++',
            #'(a++b??){2,3}+',
        )
        print(", ".join(o.data for o in objs))
        print(pat.get_string())
        pat.set_group_mode('normal')
        l = list(pat._atom._go_greedy(objs, 0))
        print(l)
        res = pat.search(objs)
        if res is not None:
            print("".join(str(x) for x in res))

        pat.set_group_mode('group')
        res = pat.search(objs)
        if res is not None:
            print("".join(str(x) for x in res))

    if False: # OK
        objs = make_test_data(
            'aaAbabcDeeFa',
        )
        print(", ".join(o.data for o in objs))
        #pat = Pattern('[upper=T data=/d|a/]')
        #pat = Pattern('[a<-data upper=T][]+[data==$a]')
        pat = Pattern('[a<-upper][]+[upper==$a]')
        pat = Pattern('[a<-upper data="d"][]*[upper==$a]')
        print(pat.get_string())
        res = pat.search(objs)
        if res is not None:
            print("".join(str(x) for x in res))

    if False: # OK
        objs, pat = make_test_data(
            'aaAbabcDeeFa',
            '(a|b)c'
        )
        print(pat.string)
        print(", ".join(o.data for o in objs))
        print(pat.get_string())
        res = pat.search(objs)
        if res is not None:
            print("".join(str(x) for x in res))

    if False: # OK
        objs = make_test_data(
            'aaAbabcDeeFa',
        )
        print(", ".join(o.data for o in objs))
        pat = Pattern('([data="f"] | [data="b"] | [upper=T] )')
        print(pat.get_string())
        res = pat.search(objs)
        if res is not None:
            print("".join(str(x) for x in res))

    if False: # OK
        objs, pat = make_test_data(
            'aaAbabaabcDeeF',
            'aab?'
        )
        print(", ".join(o.data for o in objs))
        print(pat.get_string())
        pat.set_group_mode('normal')
        for res in pat.finditer(objs, overlapping=True):
            print("".join(str(x) for x in res))

    if False: # OK
        objs, pat = make_test_data(
            'aaAbabaabcDeeF',
            #'aaababaabcdeef',
            #'a++.*',
            'a++.*f',
        )
        print(", ".join(o.data for o in objs))
        print(pat.get_string())
        pat.set_group_mode('normal')
        res = pat.fullmatch(objs)
        if res is not None:
            print("".join(str(x) for x in res))

    #if False:
    #    pat = Pattern('([foo="bar" truc=/blabla/]+[var<-attr][foo>=$var)', return_objects=True)
    #    print(pat.get_string())
    #    print(pat.search(objs))

    if False: # OK
        objs, pat = make_test_data(
            'aaAbabaabcDeeF',
            '(a+b|f.)'
        )
        print(pat.get_string())

    if False: # OK
        objs = make_test_data('aaababaabc')
        pat = Pattern('(?P<one>[data="a"][data="b"]) ([])')
        #pat = Pattern('[data="a"]')
        pat.search(objs)
        #print(pat._group_atoms['one'].start)
        group = pat.get_group(2, objs=objs)
        print(pat.get_group(1))
        print(pat.get_group(2))
        print(pat._group_atoms)
        if group is not None:
            print("".join(str(x) for x in group))

    if False: # OK
        objs = make_test_data(
            'aaab',
        )
        objs[0].content = make_test_data('ab')
        objs[1].content = make_test_data('cd')
        objs[2].content = make_test_data('cdd')
        objs[3].content = make_test_data('ccddd')
        patterns = [
            '[data="a" content={[data="c"][data="d"]}]++',
            '[data="a" content=={[data="c"][data="d"]}]++',
            '[data="a" content~{[data="c"][data="d"]}]++',
            '[data="a" content!={[data="c"][data="d"]}]++',
            '[data="a" content!=={[data="c"][data="d"]}]++',
            '[data="a" content!~{[data="c"][data="d"]}]++',
            '[data="b" content!={[data="c"][data="d"]}]++',
            '[data="b" content={[data="d"][data="d"]}]++',
            '[data=/a|b/ content~{[data="c"][data="d"]}]++',
        ]
        for pat in patterns:
            pat = Pattern(pat)
            res = pat.search(objs, return_objects=False)
            print(pat.get_string(), res)

    if False:
        objs = pd.DataFrame(
            ((x.lower(), x.isupper()) for x in 'aaAbabaabcDeeF'),
            columns=['data', 'upper']
        )
        print(objs)
        pat = make_test_pattern('aab?')
        pat.set_group_mode('normal')
        for res in pat.finditer(objs, overlapping=True):
            print(res)

    if False:
        objs = pd.DataFrame(
            ((x.lower(), x.isupper()) for x in 'aaAbabaabcDeeF'),
            columns=['data', 'upper']
        )
        print(objs)
        pat = make_test_pattern('aab?')
        pat.set_group_mode('normal')
        hl = pd.Series(False, index=objs.index)
        bio = pd.Series("O", index=objs.index)
        for res in pat.finditer(objs, overlapping=False):
            hl = hl | pd.Series(True, index=res.index)
            bio.update(pd.Series(["B"] + ["I"] * (len(res)-1),
                index=res.index))
            print(res)
        print(bio)
        print(hl)

    if False:
        objs = pd.DataFrame(
            ((x.lower(), x.isupper()) for x in 'aaAbabaabcDeeF'),
            columns=['data', 'upper']
        )
        pat = make_test_pattern('aab?')
        print(build_highlight_series(objs, pat))
        print(build_bio_series(objs, pat))

    if True:
        objs = pd.DataFrame(
            ((x.lower(), x.isupper()) for x in 'aaAbabaabcaabDeeF'),
            columns=['data', 'upper']
        )
        #                                     'aaAbabaabcaabDeeF'
        objs['chunk'] = [int(x) for x in list('11111112222222222')]
        pat = make_test_pattern('aab?')
        print(build_highlight_series(objs, pat, groupby='chunk'))
        print(build_bio_series(objs, pat, groupby='chunk'))
        for res in by_group(objs, pat, groupby='chunk', func='search'):
            print(res)


if __name__ == '__main__':
    main()
