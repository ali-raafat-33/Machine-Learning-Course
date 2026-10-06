# 🐍 Python for the ML Journey

> **Pure Python fundamentals for Machine Learning — no ML libraries required.**

This lecture introduces the core Python concepts that you will see again and again when working with Machine Learning notebooks.

The goal is not to learn Python in isolation. The examples are written around **datasets, features, labels, predictions, metrics, cross-validation, and model results** so that you can connect Python syntax directly to ML work.

---

## 🎯 Learning Objectives

By the end of this lecture, you should be able to:

- Create and work with variables and basic data types.
- Perform mathematical operations and comparisons.
- Format ML results using **f-strings**.
- Work with **lists, tuples, and dictionaries**.
- Write conditions using `if`, `elif`, and `else`.
- Use `for` loops with `range()`, `enumerate()`, and `zip()`.
- Write list comprehensions.
- Create reusable functions.
- Use default parameters and return values.
- Understand basic `lambda` functions.
- Use common Python built-in functions.
- Combine these concepts to write code similar to a real ML notebook.

---

# 1. Variables & Data Types

A **variable** is a name used to store a value.

```python
dataset = "Titanic"
rows = 891
accuracy = 0.8234
is_trained = True
```

Python has several basic data types that appear frequently in ML:

| Type | Meaning | Example |
|---|---|---|
| `int` | Whole number | `891` |
| `float` | Decimal number | `0.8234` |
| `str` | Text | `"Age"` |
| `bool` | True or False | `True` |
| `None` | No value | `None` |

### Checking the type

```python
print(type(accuracy))
# <class 'float'>
```

### Type Conversion

Python allows you to convert values between types:

```python
int("25")       # 25
float(7)        # 7.0
str(100)        # "100"
bool(0)         # False
bool(1)         # True
```

A common ML example is converting a Boolean value into `0` or `1`:

```python
flag = True
value = int(flag)

print(value)
# 1
```

---

# 2. Math & Comparisons

Python can perform the mathematical operations you need in your ML code.

```python
a = 10
b = 3

print(a + b)     # Addition
print(a - b)     # Subtraction
print(a * b)     # Multiplication
print(a / b)     # Division
print(a // b)    # Floor division
print(a % b)     # Remainder
print(a ** 2)    # Power
```

### Important Operators

| Operator | Meaning |
|---|---|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `//` | Floor division |
| `%` | Remainder |
| `**` | Power |

---

## Shorthand Assignment

Instead of:

```python
n = n + 1
```

you can write:

```python
n += 1
```

Other examples:

```python
n -= 1
n *= 2
```

---

## Comparisons

Comparison operators return either `True` or `False`.

```python
5 == 5     # True
5 != 3     # True
5 > 3      # True
5 < 3      # False
5 >= 5     # True
5 <= 4     # False
```

You can combine conditions:

```python
age = 25

age > 18 and age < 60
age < 10 or age > 80
not (age > 18)
```

Python also supports chained comparisons:

```python
18 <= age <= 60
```

---

# 3. Strings & f-Strings

A **string** is text.

```python
name = "Titanic"
```

### Why are f-strings important?

You will frequently use f-strings to display:

- Dataset information
- Accuracy
- RMSE
- Predictions
- Training results
- Model metrics

Example:

```python
dataset = "Titanic"
rows = 891
accuracy = 0.8234

print(f"Dataset: {dataset}")
print(f"Rows: {rows}")
print(f"Accuracy: {accuracy}")
```

### Formatting Numbers

```python
accuracy = 0.8234

print(f"{accuracy:.2f}")   # 0.82
print(f"{accuracy:.4f}")   # 0.8234
print(f"{accuracy:.1%}")   # 82.3%
print(f"{accuracy:.2%}")   # 82.34%
```

You can also add thousands separators:

```python
big = 1234567

print(f"{big:,}")
# 1,234,567
```

---

## Useful String Methods

```python
text = "  Hello World  "

text.strip()          # Remove spaces at the edges
text.lower()          # Convert to lowercase
text.upper()          # Convert to uppercase
text.replace("World", "ML")
```

Other useful operations:

```python
"Age" in "PassengerAge"
len("PassengerId")
```

---

# 4. Lists

A **list** stores multiple values in one variable.

Lists are extremely common in ML code.

You may use them for:

- Feature names
- Labels
- Scores
- Predictions
- Dataset columns

Example:

```python
features = ["Age", "Fare", "Pclass", "Sex"]
scores = [0.82, 0.79, 0.84, 0.81]
```

---

## Indexing

Python indexes start from **0**.

```python
features = ["Age", "Fare", "Pclass", "Sex"]

features[0]     # Age
features[1]     # Fare
features[-1]    # Sex
```

### Slicing

The syntax is:

```python
list[start:stop]
```

The `stop` index is **not included**.

```python
features[0:3]
features[:3]
features[2:]
features[::2]
```

---

## Modifying Lists

```python
features.append("Embarked")
features.remove("Fare")
features[0] = "AgeGroup"
```

Lists can also be combined:

```python
a = ["Age", "Fare"]
b = ["Sex", "Pclass"]

print(a + b)
```

---

## Useful List Functions

```python
scores = [0.82, 0.79, 0.84, 0.81, 0.83]

len(scores)
sum(scores)
min(scores)
max(scores)
sorted(scores)
```

Membership can be checked using `in`:

```python
"Age" in features
"Weight" in features
```

---

# 5. Tuples

A **tuple** is similar to a list, but it cannot be changed after it is created.

```python
shape = (891, 12)
```

Tuples are useful for values that should stay fixed, such as:

- Dataset shape
- Plot size
- RGB values
- Multiple values returned from a function

### Accessing Values

```python
shape[0]
shape[1]
```

### Unpacking

A very common Python pattern is:

```python
rows, cols = shape
```

Now:

```python
rows  # 891
cols  # 12
```

Because tuples are immutable, this is not allowed:

```python
t = (1, 2, 3)
t[0] = 99
```

---

## Returning Multiple Values

A function can return multiple values as a tuple:

```python
def min_max(numbers):
    return min(numbers), max(numbers)

low, high = min_max([5, 2, 8, 1, 9])
```

This same idea appears frequently in ML code when multiple values are returned from a function.

---

# 6. Dictionaries

A **dictionary** stores data as:

```text
key → value
```

Example:

```python
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}
```

Dictionaries are especially useful in ML for:

- Hyperparameters
- Model results
- Label mappings
- Configuration values

### Accessing Values

```python
params["n_estimators"]
params["max_depth"]
```

### Safe Access

`.get()` allows you to provide a default value:

```python
params.get("learning_rate")
params.get("learning_rate", 0.1)
```

---

## Add and Update Values

```python
params["learning_rate"] = 0.1
params["max_depth"] = 10
```

Check whether a key exists:

```python
"max_depth" in params
"gamma" in params
```

Get all keys and values:

```python
params.keys()
params.values()
```

---

## Looping Through a Dictionary

```python
results = {
    "train_accuracy": 0.9123,
    "test_accuracy": 0.8234,
    "train_rmse": 2.15,
    "test_rmse": 3.42
}

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

---

## Dictionaries for Label Mapping

A common preprocessing pattern is mapping text labels to numbers:

```python
sex_map = {
    "male": 0,
    "female": 1
}
```

Then:

```python
sex_values = ["male", "female", "male"]

encoded = [sex_map[s] for s in sex_values]

print(encoded)
# [0, 1, 0]
```

---

# 7. Conditions

Conditions allow your program to make decisions.

The basic structure is:

```python
if condition:
    ...
elif another_condition:
    ...
else:
    ...
```

Example:

```python
score = 0.84

if score >= 0.90:
    print("Excellent")
elif score >= 0.80:
    print("Good")
elif score >= 0.70:
    print("Acceptable")
else:
    print("Needs improvement")
```

> **Important:** Python uses indentation to define blocks of code. The lecture uses 4 spaces for indentation.

---

## One-Line Conditions

Python also supports a compact conditional expression:

```python
prediction = 1 if probability >= 0.5 else 0
```

This is very common when converting probabilities into predictions.

Example:

```python
prob = 0.73

prediction = 1 if prob >= 0.5 else 0
```

---

## Checking Overfitting

Conditions can also be used to inspect ML results:

```python
train_acc = 0.95
test_acc = 0.72

gap = train_acc - test_acc

status = "OVERFITTING!" if gap > 0.1 else "OK"
```

---

# 8. Loops

Loops allow you to repeat code.

## `for` Loop

```python
features = ["Age", "Fare", "Pclass", "Sex"]

for feature in features:
    print(feature)
```

---

## `range()`

`range()` is useful when you want to repeat something a specific number of times.

```python
for i in range(5):
    print(i)
```

You can also specify:

```python
range(start, stop, step)
```

Example:

```python
for i in range(0, 11, 2):
    print(i)
```

---

## `enumerate()`

`enumerate()` gives you both the **index** and the **value**.

```python
scores = [0.82, 0.79, 0.84]

for i, score in enumerate(scores):
    print(i, score)
```

This is very useful when working with things like cross-validation folds.

---

## `zip()`

`zip()` allows you to loop over multiple lists at the same time.

```python
features = ["Age", "Fare", "Pclass"]
coefs = [0.45, 0.31, -0.62]

for feature, coef in zip(features, coefs):
    print(feature, coef)
```

---

## Collecting Results

You can build a list while looping:

```python
fold_scores = []

for fold in range(5):
    score = 0.82
    fold_scores.append(score)
```

Then calculate the mean:

```python
mean = sum(fold_scores) / len(fold_scores)
```

---

# 9. List Comprehensions

A **list comprehension** is a short way to create a list using a loop.

Instead of:

```python
result = []

for n in numbers:
    result.append(n ** 0.5)
```

you can write:

```python
result = [n ** 0.5 for n in numbers]
```

### Basic Structure

```python
[expression for item in iterable]
```

---

## Filtering with a Condition

```python
scores = [82, 79, 84, 81, 83, 68, 91, 75]

high_scores = [s for s in scores if s >= 82]
```

---

## Transforming Values

```python
scaled = [s / 100 for s in scores]
```

Convert strings to lowercase:

```python
columns = ["Age", "Fare", "Pclass", "Sex"]

lower = [column.lower() for column in columns]
```

---

## ML Examples

Convert probabilities into binary predictions:

```python
probs = [0.9, 0.3, 0.7, 0.45]

predictions = [1 if p >= 0.5 else 0 for p in probs]
```

Clip values:

```python
fares = [5, 50, 512, 3, 200]

clipped = [min(f, 100) for f in fares]
```

Count correct predictions:

```python
correct = sum(
    1 for true, pred in zip(y_true, y_pred)
    if true == pred
)
```

List comprehensions are extremely common in Python-based ML code.

---

# 10. Functions

A **function** is a reusable block of code.

Instead of writing the same code multiple times, put it inside a function and call it whenever needed.

```python
def greet(name):
    print(f"Hello, {name}!")
```

Call it:

```python
greet("Jack")
greet("Rose")
```

---

## Functions with Return Values

Functions can return a result:

```python
def pct(part, total):
    return part / total * 100
```

Then:

```python
survival = pct(342, 891)
```

---

## Default Parameters

A parameter can have a default value:

```python
def display(label, value, decimals=2):
    print(f"{label}: {value:.{decimals}f}")
```

Now:

```python
display("Accuracy", 0.8234)
```

uses `2` decimal places.

You can also override it:

```python
display("Accuracy", 0.8234, 4)
```

---

## Returning Multiple Values

A function can return multiple values:

```python
def stats(numbers):
    n = len(numbers)
    mean = sum(numbers) / n

    return min(numbers), max(numbers), mean
```

Then unpack them:

```python
low, high, average = stats(scores)
```

---

## Lambda Functions

A `lambda` is a small anonymous function.

Example:

```python
square = lambda x: x ** 2

print(square(5))
```

Lambda functions can also be used with functions such as `map()` and `sorted()`.

Example:

```python
scaled = list(map(lambda x: x / 100, scores))
```

Sorting using a specific value:

```python
pairs_sorted = sorted(
    pairs,
    key=lambda x: x[1],
    reverse=True
)
```

---

# 11. Built-in Functions

Python provides many useful functions that you can use without defining them yourself.

Some important ones from this lecture are:

```python
len()
sum()
min()
max()
sorted()
reversed()
abs()
round()
```

Example:

```python
numbers = [5, 2, 8, 1, 9]

len(numbers)
sum(numbers)
min(numbers)
max(numbers)
sorted(numbers)
```

---

## `enumerate()` and `zip()`

These are also important built-ins that appear frequently in ML code:

```python
for i, column in enumerate(columns):
    print(i, column)
```

```python
for column, coef in zip(columns, coefficients):
    print(column, coef)
```

---

## `isinstance()`

`isinstance()` checks whether a value belongs to a specific type.

```python
isinstance(25, int)
isinstance("text", str)
isinstance(True, bool)
```

This can be useful when you need to inspect the type of data while your program is running.

---

# 🧩 Final Practice — Put Everything Together

The final exercise combines the concepts from the lecture into a small ML-style workflow.

The example uses:

- Variables
- f-strings
- Lists
- Dictionaries
- Conditions
- Functions
- Loops
- `enumerate()`
- Basic metric calculations

Example:

```python
dataset = "Titanic"

n_rows = 891
n_cols = 12
survived = 342

print(f"Dataset : {dataset}")
print(f"Shape   : {n_rows} x {n_cols}")
print(f"Survived: {survived} ({survived/n_rows:.1%})")
```

Feature lists:

```python
features = [
    "Age",
    "Fare",
    "Pclass",
    "Sex",
    "SibSp",
    "Parch"
]

cols_to_drop = [
    "Name",
    "Ticket",
    "Cabin",
    "PassengerId"
]
```

Model results can be stored in a dictionary:

```python
results = {
    "train": 0.9123,
    "test": 0.8234,
    "gap": 0.0889
}
```

Then use a condition:

```python
status = (
    "Overfitting!"
    if results["gap"] > 0.1
    else "Good fit"
)
```

And use a function with a loop to analyze cross-validation scores:

```python
def categorize(score):
    if score >= 0.90:
        return "Excellent"
    elif score >= 0.80:
        return "Good"
    elif score >= 0.70:
        return "Acceptable"
    else:
        return "Poor"


cv_scores = [0.82, 0.79, 0.84, 0.81, 0.83]

for i, score in enumerate(cv_scores):
    print(
        f"Fold {i+1}: "
        f"{score:.4f} "
        f"({categorize(score)})"
    )
```

---

# 📌 Quick Cheat Sheet

| Topic | Important Syntax |
|---|---|
| Variables | `x = value` |
| Types | `int`, `float`, `str`, `bool`, `None` |
| Conversion | `int()`, `float()`, `str()`, `bool()` |
| Type check | `type()` |
| Math | `+ - * / // % **` |
| Comparison | `== != > < >= <=` |
| Logic | `and`, `or`, `not` |
| f-string | `f"{x}"` |
| Formatting | `f"{x:.2f}"` |
| List | `[1, 2, 3]` |
| List indexing | `list[0]` |
| List slicing | `list[1:3]` |
| Add to list | `.append()` |
| Remove | `.remove()` |
| Tuple | `(1, 2)` |
| Dictionary | `{"key": value}` |
| Dictionary access | `d["key"]` |
| Condition | `if / elif / else` |
| Loop | `for x in items` |
| Range | `range()` |
| Index + value | `enumerate()` |
| Two lists together | `zip()` |
| List comprehension | `[x for x in items]` |
| Function | `def function():` |
| Return | `return value` |
| Lambda | `lambda x: ...` |
| Common functions | `len()`, `sum()`, `min()`, `max()`, `sorted()` |
| Type checking | `isinstance()` |

---

# 🎯 Final Takeaway

The purpose of this lecture is to build the **Python foundation you need before working deeply with Machine Learning libraries**.

You do not need to memorize every syntax immediately.

Instead, focus on understanding these patterns:

```text
Variables
   ↓
Data Types
   ↓
Lists / Tuples / Dictionaries
   ↓
Conditions
   ↓
Loops
   ↓
List Comprehensions
   ↓
Functions
   ↓
Built-in Functions
   ↓
ML-style Python Code
```

The best way to learn is simple:

> **Run every cell → Change the values → Observe the output → Experiment with your own examples.**

Once these Python fundamentals become comfortable, working with libraries such as NumPy, Pandas, Scikit-learn, PyTorch, and TensorFlow becomes much easier.
