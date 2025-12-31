import flwr as fl
from data_utils import load_dataset, non_iid_split
from client import FLClient
from server import strategy

NUM_CLIENTS = 5

# train_ds = load_dataset("data/train")
# val_ds = load_dataset("data/val")

# BASE_PATH = "140k_real_fake_face/real_vs_fake/real-vs-fake"
BASE_PATH = "data"
TRAIN_PATH = f"{BASE_PATH}/Train"
VAL_PATH = f"{BASE_PATH}/Validation"
TEST_PATH = f"{BASE_PATH}/Test"

train_ds, val_ds, test_ds = load_dataset(TRAIN_PATH, TEST_PATH, VAL_PATH)


client_train_sets = non_iid_split(train_ds, NUM_CLIENTS)
client_val_sets = non_iid_split(val_ds, NUM_CLIENTS)

def client_fn(cid):
    cid = int(cid)
    return FLClient(
        client_train_sets[cid],
        client_val_sets[cid]
    )

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=10)
)
