from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple

class BaseDataset(Dataset):
    """
    Generic base class for all datasets.
    """

    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        preload: bool = False,
    ):
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform
        self.preload = preload
        self.samples: List[Any] = []  # list of (input, target)
        self.metadata: Dict[str, Any] = {}

        self._load_metadata()

        if self.preload:
            self._preload_data()

    def _load_metadata(self):
        """
        Loads or constructs metadata (e.g. file paths, labels, splits).
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def _preload_data(self):
        """
        Optionally preload all data into memory.
        Override this for large datasets if needed.
        """
        self.samples = [self._load_sample(idx) for idx in range(len(self))]

    def _load_sample(self, idx: int) -> Tuple[Any, Any]:
        """
        Load a single sample given its index.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.preload:
            data, target = self.samples[idx]
        else:
            data, target = self._load_sample(idx)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target


