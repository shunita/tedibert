# DERT - Debiasing Embeddings by Regularization with Temporal distribution matching

Repository for submission to AAAI named: "Leveraging Temporal Trends for Training Contextual Word Embeddings to
Address Bias in Biomedical Applications".

The DERT model is trained through contra/models/gan_reg_model.py.

The various experimental results are generated through the files in contra/tests:

1. test_los_by_diags.py - Length of stay experiment.

1. test_readmission_by_diags.py - Readmission prediction experiment.

1. run_ner.py - Named entity recognition.

