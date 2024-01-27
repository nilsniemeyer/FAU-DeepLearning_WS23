import random


def run_lottery_simulation(total_runs):
    two_same_numbers_count = 0

    for _ in range(total_runs):
        number_of_slots = random.randint(5, 10)
        numbers = []

        for _ in range(number_of_slots):
            if random.random() < 5/86: # 3/86 getrennt für Swift Sneak, muss addiert werden
                number = random.randint(1, 112)  # Ziehe aus der Menge möglicher Enchantments
                numbers.append(number)

        if any(numbers.count(number) == 2 for number in numbers):
            two_same_numbers_count += 1

    overall_probability = (two_same_numbers_count / total_runs) * 100
    print(f"Zwei gleiche Verzauberungen: {overall_probability:.4f}%")
    return overall_probability

runs = 1_000_0000
probability_of_two_same_numbers = run_lottery_simulation(runs)
