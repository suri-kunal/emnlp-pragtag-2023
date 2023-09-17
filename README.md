# emnlp-pragtag-2023
In order to reproduce the results given in the paper, you need to install all libraries in the requirements.txt in Python 3.8. Once all the libraries are installed, you can go through one of the following two routes - 

1. Training and Inference -
   1. For pre-training using MLM, execute "Domain Adaptation.py"
   2. For fine-tuning sentence classification using labeled data after step 1, execute Classification.py
   3. For fine-tuning sentence classification using labeled data without step 1, execute training_wo_mlm.py
   4. To get the predictions after Step 2, execute inference_w_mlm.py
   5. To get the predictions after Step 3, execute inference_wo_mlm.py
   6. Use Word_Distribution_Analysis.ipynb to generate charts in the "PragTag 2023 - Vocabulary Analysis" paper.
   7. To obtain performance of model fined tuned on model pre-trained with MLM on out of split data, execute inference_w_mlm_cv.py
   8. To obtain performance of model fined tuned without pre-training on MLM on out of split data, execute inference_wo_mlm_cv.py

2. Inference -
   1. To get the predictions from models trained after MLM pre-training, execute inference_w_mlm.py
   2. To get the predictions from models trained without MLM pre-training, execute inference_wo_mlm.py
   3. Use Word_Distribution_Analysis.ipynb to generate charts in the "PragTag 2023 - Vocabulary Analysis" paper.
