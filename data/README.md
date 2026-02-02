# 🚀 Data Preparation

This guide provides step-by-step instructions to download, preprocess, and format the **CLEVR-Change**, **Spot-the-Diff**, and **Image-Editing-Request** datasets for training.

---

## 1. Raw Datasets Download
Download the following datasets and place them into the `./data` folder.

* **Datasets:** [[CLEVR-Change]](https://github.com/Seth-Park/RobustChangeCaptioning) | [[Image-Editing-Request]](https://github.com/airsplay/VisualRelationships) | [[Spot-the-Diff]](https://github.com/harsh19/spot-the-diff/blob/master/data/get_images.txt)
* **Special Requirement:** For **Spot-the-Diff**, we filter out invalid (e.g., empty) captions for better training. The filtered annotations can be found on Huggingface and Netdisk. 

**Initial Directory Structure:**
```bash
./data
├── clevr -> path/to/CLEVR-Change
├── edit  -> path/to/Image-Editing-Request
├── spot  -> path/to/Spot-the-Diff
├── convert_clevr.py
├── convert_edit.py
└── convert_spot.py
```

---

## 2. Preprocessed Pseudo-Sequences

Download the preprocessed pseudo-sequences (h5py files) from Huggingface or Netdisk and place them within their respective dataset directories as shown below:

### 📂 Detailed Directory View

<details>
<summary>Click to expand folder details</summary>

#### CLEVR-Change (`./clevr`)

* `CLEVR_processed/` (Contains `.h5` files)
* `images/`, `sc_images/`, `nsc_images/`
* `change_captions.json`, `splits.json`, etc.

#### Spot-the-Diff (`./spot`)

* `spot_processed/` (Contains `.h5` files)
* `annotations/`, `resized_images/`
* `get_images.txt`, `params_eps_...obj`

#### Image-Editing-Request (`./edit`)

* `edit_processed/` (Contains `.h5` files)
* `dataset/`, `images/`, `splits.json`
* `train.json`, `val.json`, `test.json`

</details>

---

## 3. Data Conversion

Run the conversion scripts located in the `./data` directory to generate the MMVid-compatible structure:

```bash
cd data

python convert_clevr.py
python convert_edit.py
python convert_spot.py
```

This will create new folders (e.g., `clevr_for_mmvid`) containing partitioned `txt` and `video` subdirectories.

---

## 4. Final Setup: Symbolic Links

To finalize the data environment without duplicating large files, manually link the preprocessed pseudo-sequence files to the specific `video` folders:

```bash
# Link CLEVR-Change sequences
cd clevr_for_mmvid/train/video
ln -s ../../../clevr/CLEVR_processed/sc_videos.h5 ./
ln -s ../../../clevr/CLEVR_processed/nsc_videos.h5 ./

# Link Spot-the-Diff sequences
cd ../../../spot_for_mmvid/train/video
ln -s ../../../spot/spot_processed/spot_videos.h5 ./

# Link Image-Editing sequences
cd ../../../edit_for_mmvid/train/video
ln -s ../../../edit/edit_processed/edit_videos.h5 ./
```

> [!IMPORTANT]
> Ensure that the relative paths in the `ln -s` commands correctly point to your local dataset directories.

---

### Verification

After completing the steps, your `./data` directory should include the `*_for_mmvid` folders, and the `video` subfolders should contain valid links to the `.h5` files.

```bash
ls -l ./clevr_for_mmvid/train/video
```
