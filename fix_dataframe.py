import pandas as pd
from datasets import load_dataset

# Load the dataset
dt_openReasoning = load_dataset("GeneralReasoning/GeneralThought-195K")
df_dt_openReasoning = pd.DataFrame(dt_openReasoning['train'])

# Check for None values
print("Before fixing:")
print(df_dt_openReasoning[['model_reasoning', 'model_answer']].isna().sum())

# Fix the None values by replacing them with empty strings
df_dt_openReasoning['model_reasoning'] = df_dt_openReasoning['model_reasoning'].fillna("")
df_dt_openReasoning['model_answer'] = df_dt_openReasoning['model_answer'].fillna("")

# Check again after fixing
print("\nAfter fixing:")
print(df_dt_openReasoning[['model_reasoning', 'model_answer']].isna().sum())

# Now create the solution column
df_dt_openReasoning['solution'] = df_dt_openReasoning.apply(
    lambda x: "<think>" + x['model_reasoning'] + "</think>" + "\n\n" + "<answer>" + x['model_answer'] + "</answer>",
    axis=1
)

# Verify it worked by checking a few examples
print("\nSample solutions:")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(df_dt_openReasoning['solution'].iloc[i][:200] + "..." if len(df_dt_openReasoning['solution'].iloc[i]) > 200 else df_dt_openReasoning['solution'].iloc[i])

print(f"\nTotal rows in dataframe: {len(df_dt_openReasoning)}")
print(f"Solution column created successfully!")

# Uncomment the line below if you want to save the fixed dataframe
# df_dt_openReasoning.to_csv('fixed_df_dt_openReasoning.csv', index=False)
