# check_five_categories.py

import pandas as pd

# Load the data
df = pd.read_csv('sorted_rot/all_papers_with_five_categories.csv')

print("=== FIVE CATEGORY ROT DISTRIBUTION ===")
print(df['final_verdict'].value_counts())
print()

print("=== SAMPLE FROM EACH CATEGORY ===")
for category in ['Very Low ROT', 'Low ROT', 'Medium ROT', 'High ROT', 'Very High ROT']:
    sample = df[df['final_verdict'] == category].head(1)
    if len(sample) > 0:
        print(f"\n{category}:")
        print(f"  Title: {sample.iloc[0]['title'][:50]}...")
        print(f"  ROT Score: {sample.iloc[0]['rot_score']:.2f}")
        print(f"  Citations: {sample.iloc[0]['citation_count']}")
        print(f"  Year: {sample.iloc[0]['publication_year']}")

print(f"\n=== SUMMARY ===")
print(f"Total papers: {len(df)}")
print(f"Categories created: {df['final_verdict'].nunique()}")
print(f"Original rot_group values: {df['rot_group'].unique()}") 