from pipeline.loader import DataLoader
from pipeline.schema import SchemaDetector
from pipeline.cleaner import DataCleaner
from pipeline.eda import EDAEngine
from pipeline.trainer import ModelTrainer
import pprint

print("\n=== Running Trainer Preparation Test ===\n")

loader = DataLoader(r"D:\AI scientist\Phase1_AutoML_EDA\data\Sample.csv")
df, meta = loader.load()

# Schema
schema = SchemaDetector(df).detect()

# Clean
cleaned_df, clean_report = DataCleaner(df, schema).clean()

# Init Trainer
trainer = ModelTrainer(cleaned_df, schema)

# Prepare data
prep = trainer.prepare_data()
print("\nPrepare Data Output:")
pprint.pprint(prep)

print("\n=== Training All Models ===\n")
results = trainer.train_all_models()
pprint.pprint(results)

print("\nBest Model:", trainer.best_model)
print("Best Score:", trainer.best_score)

print("\n=== Saving Model & Summary ===")

model_path = trainer.save_best_model("best_model.pkl")
summary = trainer.save_training_summary("training_summary.json")

print("Model saved to:", model_path)
print("Summary:")
pprint.pprint(summary)
