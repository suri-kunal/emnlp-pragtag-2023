# emnlp-pragtag-2023
Execute files in the following order - 

1. Domain Adaptation.py - For fine-tuning MLM using unlabeled data
2. Training.py / Classification.py - For fine-tuning sentence classification using labeled data after step 1
3. training_wo_mlm.py - For fine-tuning sentence classification using labeled data without step 1
4. inference_w_mlm.py - Inference after Step 2
5. inference_wo_mlm.py - Inference after Step 3
6. Use Word Clouds.ipynb to create word cloud as shown in the paper.
