import anthropic
import json

SYSTEM_PROMPT = """You are an AI agent that helps build ML pipelines on the Titanic dataset.
The dataset has these columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.
The target column is always: Survived.

Given a user instruction, return ONLY a valid JSON object with this exact structure:
{
  "drop_columns": ["list of columns to drop"],
  "keep_columns": ["list of columns to keep, excluding Survived"],
  "drop_na": true or false,
  "fill_na": "mean" or "median" or "mode" or null,
  "encode_categorical": true or false,
  "model": "LogisticRegression" or "RandomForestClassifier" or "XGBClassifier" or "DecisionTreeClassifier" or "SVC",
  "hyperparameters": {}
}

Rules:
- Use drop_columns if user says to drop specific columns
- Use keep_columns if user says to keep specific columns (never include Survived here)
- If nothing is specified about columns, leave both as empty lists
- drop_na removes rows with missing values
- fill_na fills missing values with mean/median/mode
- encode_categorical should be true if there are text columns being used
- hyperparameters can be empty or contain valid sklearn parameters
- Return ONLY the JSON, no explanation, no markdown, no extra text
"""

def parse_instructions(user_instruction, api_key):
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_instruction}
        ]
    )

    response_text = message.content[0].text.strip()

    if "```" in response_text:
        parts = response_text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except:
                continue

    return json.loads(response_text)