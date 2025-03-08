import pandas as pd
from datasets import load_dataset

# Load the dataset
print("Loading dataset...")
dt_openReasoning = load_dataset("GeneralReasoning/GeneralThought-195K")
df = pd.DataFrame(dt_openReasoning['train'])

# Display basic information
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns in the dataset: {', '.join(df.columns)}")

# Check for null values in each column
print("\nNull values count in each column:")
print(df.isna().sum())

# Display rows with null values in model_reasoning
print("\nSample rows with null values in model_reasoning:")
null_rows = df[df['model_reasoning'].isna()].head(5)
for i, row in null_rows.iterrows():
    print(f"\n--- Row {i} ---")
    for col in df.columns:
        value = row[col]
        # Truncate long values for display
        if isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        print(f"{col}: {value}")

# Display some statistics
print("\nStatistics:")
print(f"Total rows: {len(df)}")
print(f"Rows with null model_reasoning: {df['model_reasoning'].isna().sum()} ({df['model_reasoning'].isna().sum()/len(df)*100:.2f}%)")
print(f"Rows with null model_answer: {df['model_answer'].isna().sum()} ({df['model_answer'].isna().sum()/len(df)*100:.2f}%)")

# Show a few complete rows for comparison
print("\nSample complete rows:")
complete_rows = df[~df['model_reasoning'].isna()].head(3)
for i, row in complete_rows.iterrows():
    print(f"\n--- Row {i} ---")
    for col in df.columns:
        value = row[col]
        # Truncate long values for display
        if isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        print(f"{col}: {value}")
