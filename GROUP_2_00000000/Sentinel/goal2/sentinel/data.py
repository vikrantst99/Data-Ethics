"""Load the proposals data and split it into a train part and a test part (Goal 2).
PRECEDING MODEL: 1000 proposal werden in Goal 1 generiert (CSV-Output).

Die Schritte hier sind: Finds that CSV file, reads it, adds a 0/1 target column and 
makes a repeatable train/test split.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# roposal text (this is the model input)
TEXT_COLUMN = "description"
# Text label
LABEL_COLUMN = "label"
# Label value "Red Flag"
POSITIVE_LABEL = "Red Flag"
# 1 = Red Flag, 0 = Compliant
TARGET_COLUMN = "y"

# Accepted File names (by language, if required)
DATA_FILENAMES = ["proposals_1000_EN.csv", "proposals_1000.csv"]


def find_data_path(explicit=None):
    # Find CSV file and return its path.
    if explicit is not None:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError("Given data path does not exist: " + explicit)

    # Build the list of potential folders
    here = os.path.dirname(os.path.abspath(__file__)) 
    goal2 = os.path.dirname(here) 
    project = os.path.dirname(goal2)
    search_folders = [
        os.getcwd(),
        goal2,
        os.path.join(project, "GOAL 1"),
        os.path.join(goal2, "..", "GOAL 1"),
    ]

    # Go through every folder and every file name and check if the file is there.
    tried = []
    for folder in search_folders:
        for name in DATA_FILENAMES:
            candidate = os.path.normpath(os.path.join(folder, name))
            tried.append(candidate)
            if os.path.exists(candidate):
                return candidate

    message = "Proposals CSV not found. Tried:\n" + "\n".join(tried)
    raise FileNotFoundError(message)


def load_proposals(path=None):
    # Read the proposals CSV and add the number column 'y'
    # y = 1 means Red Flag (a rule violation we want to catch), y = 0 means OK
    real_path = find_data_path(path)
    df = pd.read_csv(real_path)
    if LABEL_COLUMN not in df.columns:
        raise KeyError("Expected column '" + LABEL_COLUMN + "' in " + real_path)
    df[TARGET_COLUMN] = (df[LABEL_COLUMN] == POSITIVE_LABEL).astype(int)
    return df


def make_split(df, test_size=0.20, random_state=42):
    # Split the rows into a train part and a test part.
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")
    idx_train, idx_test = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )
    return idx_train, idx_test


def get_xy(df, idx):
    # Return the text column X and the target column y for the given rows
    X = df.loc[idx, TEXT_COLUMN].fillna("")
    y = df.loc[idx, TARGET_COLUMN]
    return X, y
