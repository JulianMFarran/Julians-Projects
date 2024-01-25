import random

# List of programming topics for challenges
topics = ["Algorithms", "Data Structures", "String Manipulation", "Dynamic Programming", "Sorting"]

# Dictionary to store challenge difficulty levels and their corresponding points
difficulty_levels = {
    "Easy": 10,
    "Medium": 20,
    "Hard": 30
}

# Dictionary to store challenge explanations
challenge_explanations = {
    "Algorithms": "An algorithm is a set of instructions designed to solve a specific problem.",
    "Data Structures": "Data structures are a way of organizing and storing data to perform operations efficiently.",
    "String Manipulation": "String manipulation involves modifying or extracting parts of a string.",
    "Dynamic Programming": "Dynamic programming is an optimization technique that solves complex problems by breaking them down into simpler overlapping subproblems.",
    "Sorting": "Sorting is the process of arranging elements in a specific order."
}

# Dictionary to store challenge sample solutions
challenge_solutions = {
    "Algorithms": "Here's a sample solution for the algorithm challenge.",
    "Data Structures": "Here's a sample solution for the data structures challenge.",
    "String Manipulation": "Here's a sample solution for the string manipulation challenge.",
    "Dynamic Programming": "Here's a sample solution for the dynamic programming challenge.",
    "Sorting": "Here's a sample solution for the sorting challenge."
}

# Function to generate a random challenge
def generate_challenge():
    topic = random.choice(topics)
    difficulty = random.choice(list(difficulty_levels.keys()))
    points = difficulty_levels[difficulty]
    explanation = challenge_explanations[topic]
    solution = challenge_solutions[topic]

    print("Challenge:")
    print(f"Topic: {topic}")
    print(f"Difficulty: {difficulty}")
    print(f"Points: {points}")
    print(f"Explanation: {explanation}")
    print(f"Solution: {solution}")
    print()

# Function to track and display progress
def track_progress(score):
    print("Progress:")
    print(f"Total Score: {score}")
    print()

# Main program loop
def main():
    score = 0
    while True:
        generate_challenge()
        score += 10  # Update score based on the challenge difficulty
        track_progress(score)
        choice = input("Continue? (y/n): ")
        if choice.lower() != "y":
            break

# Run the program
if __name__ == "__main__":
    main()
