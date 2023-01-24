from os import path

TYPE = "natural"  # "cameras", "computers", "natural"
SIZE = "medium"
MODEL_TYPE = "fine_tuned"  # "fine_tuned", "pre_trained"


if TYPE == "cameras":
    TYPE_PATH = "cameras_train"

elif TYPE == "computers":
    TYPE_PATH = "computers"

elif TYPE == "natural":
    TYPE_PATH = "natural"

# paths
BASE_PATH = path.join("D:/Repos/")
REPO_PATH = path.join(BASE_PATH, "NLP-2022W-MOP/PROJECTS/mop_probing_ecomm/")

EMBEDDING_PATH = path.join(REPO_PATH, f"datasets/{TYPE_PATH}/embeddings/{MODEL_TYPE}/")

RAW_DATASET_PATH = path.join(REPO_PATH, f"datasets/{TYPE_PATH}/")
DATA_PATH = path.join(REPO_PATH, f"datasets/{TYPE_PATH}/data/")

SIMILARITY_PATH = path.join(REPO_PATH, r"similarity\30-11-2022\\")
PRETRAIN_OUTPUT_PATH = path.join(REPO_PATH, r"pretraining_output\30-11-2022\\")
DATASETS_PATH = path.join(REPO_PATH, r"datasets")
SENTEVAL_PROBING_PATH = path.join(DATASETS_PATH, r"probing")
