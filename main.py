import argparse
import os
import pandas as pd
from agent import parse_instructions
from preprocessing import preprocess
from trainer import train_and_evaluate

DATA_PATH = "titanic.csv"

def main():
    parser = argparse.ArgumentParser(description="AI-powered Titanic ML Agent")
    parser.add_argument("--infile", type=str, help="Path to .txt file with instructions")
    parser.add_argument("--outfile", type=str, help="Path to .txt file to save output")
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = input("Enter your Anthropic API key: ").strip()

    # Get instructions
    if args.infile:
        with open(args.infile, "r") as f:
            instruction = f.read().strip()
        print(f"\nInstructions from file: {instruction}")
    else:
        instruction = input("\nEnter instructions: ").strip()

    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Step 1: Parse instructions with LLM
    print("Sending instructions to Claude...")
    plan = parse_instructions(instruction, api_key)
    print(f"\nPlan from Claude: {plan}")

    # Step 2: Preprocess
    print("\nPreprocessing data...")
    df_processed = preprocess(df, plan)
    print(f"Dataset shape after preprocessing: {df_processed.shape}")

    # Step 3: Train and evaluate
    print("\nTraining model...")
    output = train_and_evaluate(df_processed, plan)

    # Print results
    print(output)

    # Save to outfile if specified
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(f"Instructions: {instruction}\n")
            f.write(f"Plan: {plan}\n")
            f.write(output)
        print(f"\nResults saved to {args.outfile}")

if __name__ == "__main__":
    main()