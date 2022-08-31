import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "../inputs/bert_base_uncased"
MODEL_PATH = "model.bin"
TRAIN_FILE_PATH = "../inputs/data/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH, do_lower_case=True
)
