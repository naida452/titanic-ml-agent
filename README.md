# Titanic ML Agent

A simple AI agent that creates, trains and evaluates a machine learning model on the Titanic dataset. The entire process is defined by a single instruction message provided by the user at runtime.

## How it works

The user provides a natural language instruction (e.g. "Drop all columns except Age and train a Random Forest model"). The instruction is sent to an LLM (Claude Haiku via Anthropic API) which returns a structured JSON plan. The script then preprocesses the data according to the plan, trains the specified ML model, and outputs evaluation metrics.

## Requirements

- Python 3.x
- anthropic
- pandas
- scikit-learn
- xgboost

## Installation

1. Clone the repository or download the files
2. Navigate to the project folder:

```
cd arti_analytics
```

3. Install dependencies:

```
pip install pandas scikit-learn xgboost anthropic
```

## Usage

Run the script:

```
python -m main
```

You will be prompted to enter your Anthropic API key and an instruction. Example instructions:

- "Drop all columns except Age and Sex, and train a Random Forest model"
- "Train a logistic regression model using all available features"
- "Drop the Cabin and Name columns, fill missing values with mean, and train a decision tree"
- "Train a xgboost model"

## Optional parameters

Use an input file instead of typing instructions:

```
python -m main --infile instructions.txt
```

Save output to a file:

```
python -m main --outfile results.txt
```

Both can be combined:

```
python -m main --infile instructions.txt --outfile results.txt
```

## Example output

```
Model: RandomForestClassifier

Confusion Matrix:
--
87  18
59  15
--
Accuracy:  56.98%
F1:        0.28
Precision: 0.45
Recall:    0.20
```

## Project structure

```
arti_analytics/
├── main.py            # Entry point, argument parsing
├── agent.py           # Communication with LLM (Claude Haiku)
├── preprocessing.py   # Data preprocessing based on LLM plan
├── trainer.py         # Model training and evaluation
├── titanic.csv        # Dataset
├── assumptions.txt    # Assumptions made during development
├── questions.txt      # Answers to additional questions
└── README.md          # This file
```
