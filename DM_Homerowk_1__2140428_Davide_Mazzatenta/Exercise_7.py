from collections import defaultdict

# Initialize dictionaries 
scores = defaultdict(float) 
reviews = defaultdict(int)
average_scores = dict()

# Read data
with open('beers.txt', 'r') as text:
    for line in text:

        # Split lines 
        data = line.strip().split('\t')
        beer_name = data[0]
        score = float(data[1]) 

        # Update dictionaries
        scores[beer_name] += score
        reviews[beer_name] += 1

# Calculate average scores for beers with at least 100 reviews
for beer_name, total_score in scores.items():
    if reviews[beer_name] >= 100:
        average_scores[beer_name] = total_score / reviews[beer_name]

# Get the top 10 beers and print the result
top_10_beers = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 Beers with the Highest Average Score and minimum 100 reviews:")
for beer_name, avg_score in top_10_beers:
    print(f"{beer_name}: {avg_score:.2f}")