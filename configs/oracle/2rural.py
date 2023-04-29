from configs.OracleToRURAL import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, TARGET_SET,TEST_DATA_CONFIG
MODEL = 'ResNet'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SAVE_PRED_EVERY = 2000

SNAPSHOT_DIR = './log/oracle/2rural'

#Hyper Paramters

WEIGHT_DECAY = 0.05
LEARNING_RATE = 0.0001
NUM_STEPS = 15000
NUM_STEPS_STOP = 10000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS / 20)
POWER = 0.9
EVAL_EVERY=2000


TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG=SOURCE_DATA_CONFIG
EVAL_DATA_CONFIG=EVAL_DATA_CONFIG
TEST_DATA_CONFIG=TEST_DATA_CONFIG