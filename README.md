# simple-does-it-utils

This is a collection of scripts that were used during my study of the Simple Does It paper on a semantic substation dataset.

## Most important scripts:

**mail.py:** Uses the Gmail API to send notification emails during training rounds.
**post_process_crf_filter_list.py:** Allows the user to select which classes to use out of the 15 originally available during the post-processing phase of SDI iterations. This was used for models trained on only reclosers, only porcelain pin insulators, and a model that had all classes except porcelain pin insulators.
**simply_doing_int_recloser_grabcut.bash:** An example of a Bash script used to apply all the code used for SDI, including model training, running detection, and post-processing the masks.
**json2png.py:** Converts masks from the VOC-style polygonal JSON format into .png masks that the DeepLabV3+ implementation used.
**segclass_maker.ipynb:** A Jupyter notebook used to create box-style segmentation masks from YOLO .txt box annotations.
**segclass_maker_boxi.ipynb:** A Jupyter notebook used to create box^i-style segmentation masks from YOLO .txt box annotations.
**segclass_maker_grabacut.ipynb:** A Jupyter notebook used to create grabacut-style segmentation masks from YOLO .txt box annotations. This implementation used box^i as a base, using the 20% innermost as definitely foreground and the remaining 80% as likely foreground.
**file_shuffler.ipynb:** An example of a Jupyter notebook used to shuffle and split datasets into training, validation, and testing subsets.

These scripts were used to process and analyze a semantic substation dataset as part of a research study.