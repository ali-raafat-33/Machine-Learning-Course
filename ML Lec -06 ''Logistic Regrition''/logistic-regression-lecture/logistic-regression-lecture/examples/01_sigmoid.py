"""Sigmoid, odds and log-odds in a few lines."""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for z in [-5, -2, 0, 2, 5]:
    print(f"z = {z:3} -> sigma(z) = {sigmoid(z):.3f}")

print("\nSymmetry: sigma(2) + sigma(-2) =", sigmoid(2) + sigmoid(-2))
print("Slope at z = 0 (= sigma*(1-sigma)):", sigmoid(0) * (1 - sigmoid(0)))

p = 0.8
odds = p / (1 - p)
print(f"\np = {p} -> odds = {odds:.1f} -> log-odds = {np.log(odds):.3f}")
print("Back to probability:", sigmoid(np.log(odds)))
