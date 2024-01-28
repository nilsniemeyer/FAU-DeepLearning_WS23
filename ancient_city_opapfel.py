import random
from collections import Counter

def run_lottery_simulation(total_runs):
    results = Counter()
    for _ in range(total_runs):
        number_of_slots = random.randint(5, 10)
        opapfel = 0

        for _ in range(number_of_slots):
            if random.random() < 1/86:
                opapfel += random.choice([1, 2])

        results[opapfel] += 1

    return results

def print_results_as_percentages(results, total_runs):
    sorted_results = sorted(results.items())
    for opapfel, count in results.items():
        percentage = (count / total_runs) * 100
        print(f"Anzahl der OP Äpfel: {opapfel}, Häufigkeit: {percentage:.4f}%")

runs = 1_000_0000
lottery_results = run_lottery_simulation(runs)
print_results_as_percentages(lottery_results, runs)
