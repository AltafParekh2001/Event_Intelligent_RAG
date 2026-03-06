import logging
import pandas as pd
from sqlalchemy import create_engine
from config import DB_PATH

logger = logging.getLogger(__name__)


def ingest_csv(csv_path: str) -> pd.DataFrame:
    """
    Read a CSV file, persist it to SQLite, and return the DataFrame.
    """
    logger.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("event_details", engine, if_exists="replace", index=False)

    logger.info(
        "CSV stored in SQLite (%s) – %d rows, %d columns",
        DB_PATH, len(df), len(df.columns),
    )
    return df
