import pandas as pd

# filepath
fn = "/wsl.localhost/Ubuntu/home/jip/nfi_germany_treedata/bwi_tree.csv"

df = pd.read_csv(fn, sep=";", encoding="iso-8859-1")

def filter_rows(df):
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

