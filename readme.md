## NER using Transformers

1. Train custom NER model continually; one on top of other on three datasets.

2. In every iteration of model train, take 100 samples of previous train set and add it to current dataset, on which the model is training.

3. In every iteration, take 20% test_set out of whole dataset and merge it with previous test_set, which makes complete "final_test_set" for model evaluation in that iteration.

4. After continual training, train a model on complete data (G1 + G2 + G3), at once, keep 20% data as test set.

5. Push each of the iterative models to huggingface-hub.

6. F1 scores of the entity_labels individually and overall for every iteration of training.

## Saved model links:

1. Link to trained Hugging Face Models:

    **T1 - https://huggingface.co/raunak6898/bert-finetuned-ner-t1** <br>
    **T2 - https://huggingface.co/raunak6898/bert-finetuned-ner-t2**
    <br>
    **T3 - https://huggingface.co/raunak6898/bert-finetuned-ner-t3**
    <br>
    **T4(all data trained together) - https://huggingface.co/raunak6898/bert-finetuned-ner-all_data**

3. CSV for metric calculated on trained model 

4. Training Code(see documented notebook - ***training_notebooks.ipynb***, kindly ignore rendering issues) and a functional script for new training and evaluation (see ***train_new.py***)

## Data Findings

* In 3 datasets, mostly clean data with no missing data, except for few entity labels index spilling over texts - Cleaned in training notebook and documented . 

* Another important finding is that, the start character for all entities are offset one to the right index, the end index is fine , so to fetch the entity text we will have to use [start_idx-1] .

## Approach

1. Custom NER training with SpaCy pipeline - notebook added ***spacy_train.ipynb*** 

2. Fine Tuned a BertForTokenClassification model, using data transformations from [text, idx]based annotations (which would have been best in spacy) to [tokens, IOB] entity tags ids within a tokenized input and labels for training using transformers Training pipeline.<br> See https://www.geeksforgeeks.org/nlp-iob-tags/ for IOB tagging.

3. Challenges with transformers training - when you move from index, text combination to IOB, tokens is that:
    * First off the transformers tokenizer is different from the tokenizer we woould have used for creating data in IOB format, which will cause mismatch in  lenght of input and labels during training.

    * Taking care of subtokens generated by transformers tokenizer by writing rules to correctly using 'I-' and 'B-' tags for new generated tokens list.

4. Trained continual learning and the (G1+G2+G3) model within a loop pipeline, iterating over model_checkpoints in and model checkpoints out being pushed to huggingface-hub, which is used in model_checkpoint_in in subsequent trainig iterations.

5. Enviornment setup:

    ```shell
    python -m venv .venv
    pip install -r requirements.txt
    ```

6. See - ***train_new.py*** for running more trainings and evaluating new iterative training models, provided with two separate functions - ***utils.py*** for utility functions.


#### Links:
 - GitHub: https://github.com/rauni-iitr
 - LinkedIn: https://www.linkedin.com/in/raunak-7068/








