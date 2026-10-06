# Object-Oriented Programming (OOP) in Python

## 📚 Lecture Overview

Object-Oriented Programming (OOP) is a programming style that organizes code around **objects and classes**.

Instead of writing a large program as a collection of functions, OOP allows us to represent real-world things as **objects** that contain:

* **Data** → Attributes
* **Behavior** → Methods

OOP helps us write code that is more **organized, reusable, maintainable, and scalable**.

---

## 🎯 Learning Objectives

After completing this lecture, you should be able to:

* Understand what **Classes** and **Objects** are.
* Create and initialize objects using `__init__()`.
* Work with **Attributes** and **Methods**.
* Understand the four main pillars of OOP:

  * Encapsulation
  * Inheritance
  * Polymorphism
  * Abstraction
* Use `super()` to work with parent classes.
* Build a simple real-world system using OOP.

---

# 1. Introduction to OOP

## What is OOP?

OOP stands for **Object-Oriented Programming**.

It is a programming paradigm that organizes code around **objects and classes** instead of only using functions and procedures.

For example, think about a **Car**.

A car has:

### Data

* Brand
* Speed
* Color

### Behavior

* Start
* Stop
* Accelerate

In OOP, we can represent the car as an **object**.

### Why do we use OOP?

OOP provides several important benefits:

| Benefit             | Meaning                                     |
| ------------------- | ------------------------------------------- |
| Reusability         | Write code once and reuse it                |
| Modularity          | Divide a large problem into smaller objects |
| Maintainability     | Make code easier to update                  |
| Scalability         | Make projects easier to grow                |
| Real-world modeling | Represent real-world things using objects   |

---

# 2. Classes and Objects

## What is a Class?

A **class** is like a blueprint or template.

For example, an architect can create a blueprint for a house.

The blueprint is not the actual house.

It describes how the house should be built.

In the same way:

```python
class Car:
    brand = "Toyota"
    speed = 0
```

`Car` is the class.

---

## What is an Object?

An **object** is an actual instance created from a class.

```python
car1 = Car()
car2 = Car()
```

Here:

* `Car` → Class
* `car1` → Object
* `car2` → Object

We can give each object different values:

```python
car1.brand = "BMW"
car2.brand = "Tesla"

print(car1.brand)
print(car2.brand)
```

Output:

```text
BMW
Tesla
```

Both objects come from the same class, but each object can have its own state.

### Simple idea

```text
Class
  ↓
Blueprint

Object
  ↓
Actual instance
```

---

# 3. Constructor `__init__()`

When we create an object, we often want to give it some initial data.

Python provides a special method called:

```python
__init__()
```

This is called the **constructor**.

It runs automatically when an object is created.

Example:

```python
class Person:

    def __init__(self, name, age):
        self.name = name
        self.age = age
```

Now we can create objects:

```python
p1 = Person("Alice", 25)
p2 = Person("Bob", 30)
```

Each object receives its own data.

```python
print(p1.name)
print(p2.name)
```

Output:

```text
Alice
Bob
```

---

## What is `self`?

`self` refers to the **current object**.

For example:

```python
self.name = name
```

means:

> Store the value of `name` inside the current object.

So when we write:

```python
p1 = Person("Alice", 25)
```

Python stores:

```text
p1.name → Alice
p1.age  → 25
```

And:

```python
p2 = Person("Bob", 30)
```

stores:

```text
p2.name → Bob
p2.age  → 30
```

---

# 4. Attributes and Methods

A class usually contains two important things:

## Attributes

**Attributes are data stored inside an object.**

Example:

```python
self.owner
self.balance
```

## Methods

**Methods are functions defined inside a class.**

Example:

```python
def deposit(self, amount):
    self.balance += amount
```

A method describes what an object can **do**.

---

## Example: Bank Account

```python
class BankAccount:

    bank = "Python Bank"

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def show(self):
        print(f"{self.owner}: ${self.balance}")
```

Create an account:

```python
acc = BankAccount("Alice", 100)
```

Deposit money:

```python
acc.deposit(50)
```

Show the account:

```python
acc.show()
```

Output:

```text
Alice: $150
```

### Important distinction

```text
Attribute → Data
Method    → Behavior
```

---

# 5. Encapsulation

**Encapsulation** means controlling access to the internal data of an object.

The goal is to protect data and prevent unwanted or accidental changes.

Python commonly uses:

```text
public
_protected
__private
```

---

## Public

A public attribute can be accessed normally.

```python
self.name = name
```

---

## Protected

A single underscore is commonly used to indicate that something is intended for internal use:

```python
self._salary
```

This is mainly a convention in Python.

---

## Private

A double underscore is used for private attributes:

```python
self.__salary = salary
```

Example:

```python
class Employee:

    def __init__(self, name, salary):
        self.name = name
        self.__salary = salary

    def get_salary(self):
        return self.__salary

    def set_salary(self, amount):
        if amount > 0:
            self.__salary = amount
```

Now:

```python
emp = Employee("Alice", 5000)

print(emp.get_salary())
```

Output:

```text
5000
```

We can update the salary using:

```python
emp.set_salary(6000)
```

But trying:

```python
emp.__salary
```

results in an `AttributeError`.

### Why is Encapsulation useful?

It allows us to control **how data is accessed or changed**.

For example, the setter can validate the value:

```python
if amount > 0:
    self.__salary = amount
```

This prevents invalid values.

---

# 6. Inheritance

**Inheritance** allows one class to reuse code from another class.

We have:

```text
Parent Class
     ↓
Child Class
```

Example:

```python
class Animal:

    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name} makes a sound")
```

Now we create a child class:

```python
class Dog(Animal):

    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed
```

`Dog` inherits from `Animal`.

Therefore, `Dog` can use the functionality of `Animal`.

---

## What is `super()`?

`super()` allows us to access functionality from the parent class.

```python
super().__init__(name)
```

This calls the parent's constructor.

---

## Method Overriding

A child class can change the behavior of a method inherited from the parent.

```python
class Dog(Animal):

    def speak(self):
        print(f"{self.name} says: Woof!")
```

Now:

```python
d = Dog("Rex", "Labrador")

d.speak()
```

Output:

```text
Rex says: Woof!
```

This is called **method overriding**.

### Main benefit of inheritance

Inheritance promotes:

> **Code Reuse**

Instead of writing the same code again, we can reuse code from the parent class.

---

# 7. Polymorphism

The word **Polymorphism** means:

> Many forms.

In OOP, polymorphism allows the **same method name** to behave differently depending on the object.

For example:

```python
class Circle:

    def area(self):
        return 3.14 * self.r ** 2
```

And:

```python
class Rectangle:

    def area(self):
        return self.w * self.h
```

Both classes have:

```python
area()
```

But the implementation is different.

We can use them in the same loop:

```python
shapes = [Circle(5), Rectangle(4, 6)]

for s in shapes:
    print(s.area())
```

The same method call:

```python
s.area()
```

produces different results depending on the object.

### Simple idea

```text
Same method
     ↓
Different objects
     ↓
Different behavior
```

Python also supports polymorphism through **duck typing**.

---

# 8. Abstraction

**Abstraction** means hiding unnecessary implementation details and defining what a class should do.

Python provides the `abc` module for creating abstract classes.

```python
from abc import ABC, abstractmethod
```

Example:

```python
class Vehicle(ABC):

    @abstractmethod
    def start_engine(self):
        pass

    @abstractmethod
    def stop_engine(self):
        pass
```

Here, `Vehicle` defines a contract.

Any child class must implement:

```python
start_engine()
stop_engine()
```

---

## Implementing the Abstract Class

```python
class Car(Vehicle):

    def start_engine(self):
        print("Car engine started")

    def stop_engine(self):
        print("Car engine stopped")
```

Now:

```python
my_car = Car()

my_car.start_engine()
```

Output:

```text
Car engine started
```

But we cannot directly create:

```python
Vehicle()
```

because `Vehicle` is an abstract class.

### Main idea

Abstraction tells us:

> **What must be done, without focusing on how it is done.**

---

# 9. The Four Pillars of OOP

The four main pillars are:

## 1. Encapsulation

**Hide and protect internal data.**

```text
Protect the data
```

## 2. Inheritance

**Reuse code from another class.**

```text
Reuse the code
```

## 3. Polymorphism

**One interface, different behaviors.**

```text
Same method → Different behavior
```

## 4. Abstraction

**Hide complexity and define required behavior.**

```text
Show what is needed
Hide unnecessary details
```

---

# 10. Mini Project — Library System

The lecture combines all OOP concepts in a simple **Library System**.

The main class is:

```python
class LibraryItem(ABC):
```

It uses **Abstraction**.

It also contains:

```python
self.__year
```

which demonstrates **Encapsulation**.

Then we create:

```python
class Book(LibraryItem):
```

and:

```python
class Magazine(LibraryItem):
```

Both classes use **Inheritance**.

Each class implements:

```python
display_info()
```

in its own way.

This demonstrates **Polymorphism**.

---

## Complete Example

```python
from abc import ABC, abstractmethod


class LibraryItem(ABC):

    def __init__(self, title, year):
        self.title = title
        self.__year = year

    def get_year(self):
        return self.__year

    @abstractmethod
    def display_info(self):
        pass


class Book(LibraryItem):

    def __init__(self, title, year, author):
        super().__init__(title, year)
        self.author = author

    def display_info(self):
        print(
            f"[Book] {self.title} "
            f"by {self.author} "
            f"({self.get_year()})"
        )


class Magazine(LibraryItem):

    def __init__(self, title, year, issue):
        super().__init__(title, year)
        self.issue = issue

    def display_info(self):
        print(
            f"[Magazine] {self.title} "
            f"Issue #{self.issue} "
            f"({self.get_year()})"
        )


library = [
    Book("Python 101", 2022, "John"),
    Magazine("Tech Today", 2024, 42)
]

for item in library:
    item.display_info()
```

Output:

```text
[Book] Python 101 by John (2022)
[Magazine] Tech Today Issue #42 (2024)
```

---

# 🧠 How the Mini Project Uses OOP

| OOP Concept   | Where is it used?                          |
| ------------- | ------------------------------------------ |
| Class         | `LibraryItem`, `Book`, `Magazine`          |
| Object        | `Book(...)`, `Magazine(...)`               |
| Constructor   | `__init__()`                               |
| Attribute     | `title`, `author`, `issue`                 |
| Method        | `display_info()`                           |
| Encapsulation | `__year`                                   |
| Inheritance   | `Book(LibraryItem)`                        |
| `super()`     | Calls the parent constructor               |
| Polymorphism  | Different `display_info()` implementations |
| Abstraction   | `LibraryItem(ABC)` and `@abstractmethod`   |

---

# 📌 Quick Revision

Remember these simple definitions:

```text
Class
→ Blueprint for creating objects

Object
→ An instance of a class

Constructor
→ __init__() initializes an object

Attribute
→ Data stored in an object

Method
→ Function inside a class

Encapsulation
→ Protect internal data

Inheritance
→ Reuse code from a parent class

Polymorphism
→ Same interface, different behavior

Abstraction
→ Hide complexity and define required behavior
```

---

# 🎯 What You Should Be Able to Do

After this lecture, you should be comfortable with:

* Creating a class.
* Creating objects from a class.
* Using `__init__()`.
* Understanding `self`.
* Creating attributes and methods.
* Understanding public, protected, and private attributes.
* Creating child classes using inheritance.
* Using `super()`.
* Overriding methods.
* Understanding polymorphism.
* Creating abstract classes.
* Using `ABC` and `@abstractmethod`.
* Combining OOP concepts in a small project.

---

# 🚀 Practice

Try building your own OOP project.

Some ideas:

* 🏦 Bank Management System
* 🎓 Student Management System
* 🛒 Shopping Cart
* 🏥 Hospital Management System
* 📚 Library Management System
* 🚗 Vehicle Management System

Start with simple classes, then gradually add:

```text
Classes
   ↓
Objects
   ↓
Attributes & Methods
   ↓
Encapsulation
   ↓
Inheritance
   ↓
Polymorphism
   ↓
Abstraction
   ↓
Complete Project
```

---

## 📖 Final Takeaway

OOP is not just about writing classes.

The main goal is to organize your program into **well-structured objects** that are easier to understand, reuse, maintain, and expand.

The four most important concepts to remember are:

> **Encapsulation → Protect data**
> **Inheritance → Reuse code**
> **Polymorphism → Different behavior**
> **Abstraction → Hide complexity**

Keep practicing by converting real-world problems into classes and objects. That is the best way to become comfortable with OOP.
