from abc import abstractmethod


class TrainDataManager:
    @abstractmethod
    def calculate_nnue_input_layer(self, fen: str):
        pass

    @abstractmethod
    def get_packed_size(self):
        pass

    @abstractmethod
    def get_record_size(self):
        pass

    @abstractmethod
    def parse_record(self, record: bytes):
        pass

    @abstractmethod
    def get_input_size(self):
        pass