from pathlib import Path

readme = r'''# 🐍 Python for the ML Journey

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
