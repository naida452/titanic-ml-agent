import pandas as pd

def preprocess(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    df = df.copy()
    
    # Keep only specific columns if specified
    if plan.get("keep_columns"):
        cols = plan["keep_columns"] + ["Survived"]
        df = df[cols]
    
    # Drop specific columns if specified
    if plan.get("drop_columns"):
        df = df.drop(columns=plan["drop_columns"], errors="ignore")
    
    # Always drop columns that can't be used for ML
    always_drop = ["Name", "Ticket", "Cabin", "PassengerId"]
    df = df.drop(columns=[c for c in always_drop if c in df.columns], errors="ignore")
    
    # Handle missing values
    if plan.get("drop_na"):
        df = df.dropna()
    elif plan.get("fill_na"):
        strategy = plan["fill_na"]
        for col in df.columns:
            if df[col].isnull().any():
                if strategy == "mean" and df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median" and df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
    else:
        # Default: fill numeric with median, drop rows where Survived is missing
        df = df.dropna(subset=["Survived"])
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    
    # Always encode all text columns
    for col in df.columns:
        if df[col].dtype == "object" and col != "Survived":
            df[col] = pd.Categorical(df[col]).codes
    
    return df