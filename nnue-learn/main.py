from torch import tensor, float32
from torch.utils.data import DataLoader
from src.model.chess_dataset import ChessDataset
from src.model.train_data_manager import TrainDataManager
from src.networks.halfkp.data_manager import HalfKPDataManager
from src.networks.halfkp.network import HalfKPNetwork
from src.networks.simple.data_manager import SimpleNetworkDataManager
from src.networks.simple.network import SimpleNetwork
from src.train import train

def evaluate_position_simple(fen):
    nnue = SimpleNetwork(128)
    manager = SimpleNetworkDataManager()
    nnue.load_weights(16, "simple")
    nnue.eval()
    nnue_input = manager.calculate_nnue_input_layer(fen)
    nnue_input = tensor(nnue_input, dtype=float32)
    print(nnue(nnue_input).item())

def evaluate_position_halfkp(fen):
    nnue = HalfKPNetwork(128)
    manager = HalfKPDataManager()
    nnue.load_weights(1, "halfkp")
    nnue.eval()
    nnue_input1, nnue_input2 = manager.calculate_nnue_input_layer(fen)
    nnue_input1 = tensor(nnue_input1, dtype=float32)
    nnue_input2 = tensor(nnue_input2, dtype=float32)
    nnue_input1 = nnue_input1.unsqueeze(0)
    nnue_input2 = nnue_input2.unsqueeze(0)
    print(nnue(nnue_input1, nnue_input2).item())

def create_singlethreaded_data_loader(manager: TrainDataManager, path: str):
    dataset = ChessDataset(path, manager)
    return DataLoader(dataset, batch_size=512, num_workers=0)

def create_data_loader(manager: TrainDataManager, path: str):
    # return create_singlethreaded_data_loader(manager, path)
    dataset = ChessDataset(path, manager)
    return DataLoader(dataset, batch_size=512, num_workers=11, persistent_workers=True, prefetch_factor=2)


def run_simple_train_nnue(
        hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
        train_directory
):
    # evaluate_position_simple("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")
    # return None

    manager = SimpleNetworkDataManager()

    train(
        SimpleNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path),
        create_data_loader(manager, validation_dataset_path),
        train_directory
    )

def run_halfkp_train_nnue(
        hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
        train_directory
):
    manager = HalfKPDataManager()

    train(
        HalfKPNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path),
        create_data_loader(manager, validation_dataset_path),
        train_directory
    )


SHOULD_TRAIN_SIMPLE = False
SHOULD_TRAIN_HALFKP = True

if __name__ == '__main__':
    if SHOULD_TRAIN_SIMPLE:
        run_simple_train_nnue(
            128,
            "train.csv",
            "validate.csv",
            "simple"
        )

    if SHOULD_TRAIN_HALFKP:
        run_halfkp_train_nnue(
            128,
            "train_100millions_dataset.csv",
            "validate_100millions_dataset.csv",
            "halfkp"
        )
