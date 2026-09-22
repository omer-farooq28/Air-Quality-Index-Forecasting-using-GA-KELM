"""
Modern GA-KELM regressor compatible with Python 3.11.

The model uses an RBF kernel and a Genetic Algorithm to optimize
the KELM hyperparameters gamma and C using a validation set.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class GeneticELMRegressor(BaseEstimator, RegressorMixin):
    """Genetic Algorithm optimized Kernel Extreme Learning Machine."""

    def __init__(
        self,
        population_size=12,
        generations=15,
        mutation_rate=0.20,
        random_state=42,
        validation_size=0.20,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.validation_size = validation_size

    @staticmethod
    def _rbf_kernel(X1, X2, gamma):
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
        distances = np.maximum(X1_sq + X2_sq - 2 * X1 @ X2.T, 0.0)
        return np.exp(-gamma * distances)

    @staticmethod
    def _fit_kernel(X, y, gamma, C):
        K = GeneticELMRegressor._rbf_kernel(X, X, gamma)
        n = K.shape[0]
        reg = np.eye(n) / C
        alpha = np.linalg.solve(K + reg, y)
        return alpha

    def _fitness(self, params, X_train, y_train, X_val, y_val):
        gamma = 10.0 ** params[0]
        C = 10.0 ** params[1]
        try:
            alpha = self._fit_kernel(X_train, y_train, gamma, C)
            K_val = self._rbf_kernel(X_val, X_train, gamma)
            prediction = K_val @ alpha
            return float(np.sqrt(mean_squared_error(y_val, prediction)))
        except np.linalg.LinAlgError:
            return np.inf

    def _initial_population(self, rng):
        return rng.uniform(
            low=[-3.0, 0.0],
            high=[2.0, 4.0],
            size=(self.population_size, 2),
        )

    def _mutate(self, individual, rng):
        if rng.random() < self.mutation_rate:
            individual[0] += rng.normal(0, 0.35)
        if rng.random() < self.mutation_rate:
            individual[1] += rng.normal(0, 0.50)
        individual[0] = np.clip(individual[0], -3.0, 2.0)
        individual[1] = np.clip(individual[1], 0.0, 4.0)
        return individual

    def _evolve(self, population, fitness, rng):
        order = np.argsort(fitness)
        elite_count = max(2, self.population_size // 4)
        elites = population[order[:elite_count]].copy()

        new_population = [elite.copy() for elite in elites]

        while len(new_population) < self.population_size:
            p1 = elites[rng.integers(0, elite_count)]
            p2 = elites[rng.integers(0, elite_count)]
            mask = rng.random(2) < 0.5
            child = np.where(mask, p1, p2).copy()
            child = self._mutate(child, rng)
            new_population.append(child)

        return np.asarray(new_population)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        rng = np.random.default_rng(self.random_state)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_size,
            random_state=self.random_state,
        )

        population = self._initial_population(rng)
        best_params = None
        best_fitness = np.inf

        for _ in range(self.generations):
            fitness = np.array([
                self._fitness(individual, X_train, y_train, X_val, y_val)
                for individual in population
            ])

            index = np.argmin(fitness)
            if fitness[index] < best_fitness:
                best_fitness = fitness[index]
                best_params = population[index].copy()

            population = self._evolve(population, fitness, rng)

        self.gamma_ = 10.0 ** best_params[0]
        self.C_ = 10.0 ** best_params[1]
        self.alpha_ = self._fit_kernel(X, y, self.gamma_, self.C_)
        self.X_train_ = X
        self.best_validation_rmse_ = best_fitness
        self.fitted_ = True

        return self

    def predict(self, X):
        if not getattr(self, "fitted_", False):
            raise ValueError("GeneticELMRegressor must be fitted before prediction.")

        X = np.asarray(X, dtype=float)
        K = self._rbf_kernel(X, self.X_train_, self.gamma_)
        return (K @ self.alpha_).ravel()
