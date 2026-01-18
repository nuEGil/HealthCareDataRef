import os 
import csv
import glob
import numpy as np 
import pandas as pd


def organize_patch_paths(set_= 'mass1'):
    outpath = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'],f'user_meta_data/patch_sets/{set_}')
    fnames = glob.glob(os.path.join(outpath, '*/*.png'))  
    csv_out = os.path.join(outpath, "patch_files.csv")

    with open(csv_out, "w", newline="") as ff:
        writer = csv.writer(ff)
        writer.writerow(["filename", "label"])

        for f in fnames:
            label = os.path.splitext(f)[0].split("label-")[-1]
            writer.writerow([f, int(label)])

def split_data(set_= 'mass1'):
    outpath = os.path.join(os.environ['CHESTXRAY8_BASE_DIR'], f'user_meta_data/patch_sets/{set_}')   
    csv_out = os.path.join(outpath, "patch_files.csv")
    
    df = pd.read_csv(csv_out)

    # image_ind = everything before "_x-"
    df["image_ind"] = (
    df["filename"]
    .str.split("/")           # split path
    .str[-1]                  # basename
    .str.split("_")           # split parts
    .str[:2]                  # img_inds
    .str.join("_")            # rejoin
)

    print(df.head())
    print(df.image_ind.unique())
    
    # random number generator
    rng = np.random.default_rng(42)
    img_inds = df.image_ind.unique()

    rng.shuffle(img_inds)

    n_train = int(0.8 * len(img_inds))
    train_imgs = img_inds[:n_train]
    test_imgs  = img_inds[n_train:]

    train_rows = []
    test_rows = []

    for img in train_imgs:
        rows = df[df.image_ind == img]
        train_rows.append(rows)

    for img in test_imgs:
        rows = df[df.image_ind == img]
        test_rows.append(rows)

    train_df = pd.concat(train_rows)
    test_df  = pd.concat(test_rows)

    #Take all non background patches to be positive label, background is neg
    # will still have an imbalance.  -- change back to !=0 for all labels. 
    train_pos = train_df[train_df.label == 3]
    train_neg = train_df[train_df.label == 0].sample(
        n=len(train_pos), random_state=42
    )

    test_pos = test_df[test_df.label == 3 ]
    test_neg = test_df[test_df.label == 0].sample(
        n=len(test_pos), random_state=42
    )

    train_df = pd.concat([train_pos, train_pos, train_neg]).sample(frac=1, random_state=42)
    test_df  = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

    train_df.to_csv(os.path.join(outpath, "mass_train_set.csv"))
    test_df.to_csv(os.path.join(outpath, "mass_test_set.csv"))

    assert set(train_df.image_ind).isdisjoint(test_df.image_ind)

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--i", type=int, required=False)
    args = p.parse_args()
    # organize_patch_paths(set_= 'NoF_Eff_Inf_Mas')
    split_data(set_ = 'NoF_Eff_Inf_Mas')