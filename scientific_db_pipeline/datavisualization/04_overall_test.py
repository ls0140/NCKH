print("DETAILED FEATURE DISTRIBUTION:")
print("="*50)

for feature in ['mentions_dataset', 'mentions_metrics', 'has_github_link']:
    print(f"\n{feature}:")
    print(df.groupby(['rot_group', feature]).size().unstack(fill_value=0))
    print(f"Overall: {df[feature].sum()}/{len(df)} ({df[feature].mean()*100:.1f}%)")