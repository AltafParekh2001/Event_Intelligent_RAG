import pandas as pd


def create_event_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ALL columns of each row into a rich text block so that
    every field is embedded and searchable — not just 10 hardcoded ones.
    """
    texts = []

    for _, row in df.iterrows():
        # Build one line per non-null column: "Column Name: value"
        lines = []
        for col in df.columns:
            value = row[col]
            # Skip nulls and empty strings
            if pd.notna(value) and str(value).strip() not in ("", "nan", "None"):
                lines.append(f"{col}: {value}")

        texts.append("\n".join(lines))

    df = df.copy()
    df["event_text"] = texts
    return df