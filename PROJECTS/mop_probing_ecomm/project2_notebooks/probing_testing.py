# setting path
import glob
import os
import sys

current_dir = os.path.abspath("")
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from source.probing.testing import test_visualize_probing_task
from source.utils.probing_tasks_utils import *
from config import BASE_PATH, REPO_PATH

TYPES = ["computers", "cameras", "natural"]
MODEL_TYPES = ["pre_trained", "fine_tuned"]
SIZE = "medium"

for TYPE in TYPES:
    for MODEL_TYPE in MODEL_TYPES:
        print(TYPE, MODEL_TYPE)

        if TYPE == "cameras":
            TYPE_PATH = "cameras_train"

        elif TYPE == "computers":
            TYPE_PATH = "computers"

        elif TYPE == "natural":
            TYPE_PATH = "natural"

        EMBEDDING_PATH_PROBING = os.path.join(
            REPO_PATH,
            f"datasets/{TYPE_PATH}/embeddings/{MODEL_TYPE}/embeddings_for_probing_task_input",
        )

        DATA_PATH = os.path.join(REPO_PATH, f"datasets/{TYPE_PATH}/data/")

        os.chdir(EMBEDDING_PATH_PROBING)
        probing_task_files = glob.glob("*.{}".format("csv"))

        for file in probing_task_files:
            print(file)
            file_name = file.split(".")[0]

            file_name = file_name[11:]
            test_visualize_probing_task(
                file,
                f"{file_name}_{MODEL_TYPE}",
                EMBEDDING_PATH_PROBING,
                REPO_PATH,
                TYPE,
                SIZE,
                MODEL_TYPE,
            )

