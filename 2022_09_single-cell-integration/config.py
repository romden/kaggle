import os

SEED = 242

PATH_DATA = '/home/datashare/datasets/kaggle/input/open-problems-multimodal'
PATH_WORKING = '/home/datashare/datasets/kaggle/working/open-problems-multimodal'

FP_CELL_METADATA = os.path.join(PATH_DATA, "metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(PATH_DATA, "train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(PATH_DATA, "train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(PATH_DATA, "test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(PATH_DATA, "train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(PATH_DATA, "train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(PATH_DATA, "test_multi_inputs.h5")
FP_MULTIOME_TEST_INPUTS_reduced = os.path.join(PATH_DATA, "test_multi_inputs_reduced.h5")

FP_SUBMISSION = os.path.join(PATH_DATA, "sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(PATH_DATA, "evaluation_ids.csv")

SHAPES = {'cite': (48663, 140), 'multi': (16780, 23418)}