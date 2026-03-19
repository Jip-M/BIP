import matplotlib.pyplot as plt
import pandas as pd

# filepath
fn = "/home/jip/nfi_germany_treedata/bwi_tree.csv"

df = pd.read_csv(fn, sep=";", encoding="iso-8859-1", decimal=",")


def filter_rows(df):
    """After filtering, 208544 treeID's will remain twice: 1 time around 2000, 1 time around 2010"""
    # 1. Convert year to integer
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    # 2. Filter out samples < 1990
    df = df[df["year"] >= 1990].copy()

    # 3. Create boolean masks for the two target periods
    mask_period_1 = (df["year"] >= 1995) & (df["year"] <= 2005)
    mask_period_2 = df["year"] >= 2010

    # 4. Get the sets of Tree IDs that exist in each period
    ids_in_p1 = set(df.loc[mask_period_1, "tree_ID"])
    ids_in_p2 = set(df.loc[mask_period_2, "tree_ID"])

    # 5. Find the intersection (Trees that are in BOTH)
    valid_tree_ids = ids_in_p1.intersection(ids_in_p2)

    # 6. Filter the dataframe to only include these trees
    # and only the measurements from these two periods
    df_filtered = df[
        df["tree_ID"].isin(valid_tree_ids) & (mask_period_1 | mask_period_2)
    ].copy()

    print(f"Original trees: {df['tree_ID'].nunique()}")
    print(f"Filtered trees: {df_filtered['tree_ID'].nunique()}")

    return df_filtered


# dont need this. Plot is already saved.
def plot_barplot(df_filtered):
    year_counts = df_filtered["year"].value_counts().sort_index()

    plt.bar(year_counts.index, year_counts.values)
    plt.xlabel("Year")
    plt.ylabel("Number of measurements")
    plt.title("Filtered tree Measurements per year")

    plt.show()


def merge_columns(df_filtered):
    """splits 2000 / 2010 trees, merges 2010 tree heights with 2000's trees as 'future_height', and reorders columns"""

    # 1. Create the Feature Set (Train) from the 1995-2005 window
    train_df = df_filtered[df_filtered["year"] < 2005]

    # # 2. Create the Label Set from the 2010+ window
    label_df = df_filtered[df_filtered["year"] > 2005][["tree_ID", "height"]]

    # 1. merge columns
    label_df = label_df.rename(columns={"height": "future_height"})
    train_df = pd.merge(train_df, label_df, on="tree_ID")
    print(f"Final training rows: {len(train_df)}")

    # reorder columns
    cols = list(train_df.columns)
    leading_cols = ["tree_ID", "height", "future_height"]
    remaining_cols = [c for c in cols if c not in leading_cols]
    new_order = leading_cols + remaining_cols
    train_df = train_df[new_order]

    print("Columns successfully merged and reordered:\n")
    print(train_df.head())

    return train_df


def drop_columns(df, dropcols: list[str]):
    df = df.drop(dropcols, axis=1)
    return df


dropcols = [
    "species_type",
    "species",
    "dbh_class",
    "age_class",
    "NFI_period",
    "year",
    "previous_year",
    "time_interval",
    "ba",
    "ba_incr_y",
    "%ba_incr_y",
    "height_method",
    "tree_ID",
    "measurement_ID",
]

df_filtered = filter_rows(df)
merged_df = merge_columns(df_filtered)
print(merged_df.head())

train_df = drop_columns(merged_df, dropcols)
train_df.to_csv("preprocessed_train.csv", index=False)

print("preprocessing successful!")
