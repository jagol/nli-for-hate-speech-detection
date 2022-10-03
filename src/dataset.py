import json
import os
import csv
from typing import Optional, Dict, List, Iterator, Any


class Dataset:

    labels_str_to_num = None
    labels_num_to_str = None

    def __init__(self, path: str, name: str, **kwargs: Optional[Dict[str, str]]):
        """Initializer function.

        Args:
            path: Path to test set file if the test set is one file or
                otherwise to the directory containing the test set files.
            name: Name of the data set.
            kwargs: Any additional keyword arguments.
        """
        self.path = path
        self.name = name
        self.kwargs = kwargs
        self._items = []

    def load(self) -> List[Dict[str, str]]:
        raise NotImplementedError

    def get_num_items(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Dict[str, str]]:
        for item in self._items:
            yield item

    def get_labels_str_to_num(self) -> Dict[str, int]:
        """Return a dict mapping string to numeric labels."""
        if 'task_level' in self.kwargs:
            return self.labels_str_to_num[self.kwargs['task_level']]
        else:
            return self.labels_str_to_num

    def get_labels_num_to_str(self) -> Dict[int, str]:
        """Return a dict mapping numeric to string labels."""
        if 'task_level' in self.kwargs:
            return self.labels_num_to_str[self.kwargs['task_level']]
        else:
            return self.labels_num_to_str

    def get_str_label(self, num_label: int) -> str:
        """Given a numeric label, get the corresponding string label."""
        if 'task_level' in self.kwargs:
            return self.labels_num_to_str[self.kwargs['task_level']][num_label]
        else:
            return self.labels_num_to_str[num_label]

    def get_num_label(self, str_label: int) -> int:
        """Given a string label, get the corresponding numeric label."""
        if 'task_level' in self.kwargs:
            return self.labels_str_to_num[self.kwargs['task_level']][str_label]
        else:
            return self.labels_str_to_num[str_label]


class HateCheckDataset(Dataset):

    labels_str_to_num = {
        'hateful': 1,
        'non-hateful': 0
    }
    labels_num_to_str = {v: k for k, v in labels_str_to_num.items()}

    def load(self) -> None:
        with open(self.path) as fin:
            reader = csv.DictReader(fin)
            for i, row in enumerate(reader):
                self._items.append({
                    'id': i,
                    'text': row['test_case'],
                    'label': row['label_gold'],
                    'category': row['functionality']
                })

    def get_num_items(self) -> int:
        count = 0
        with open(self.path) as fin:
            reader = csv.DictReader(fin)
            for _ in reader:
                count += 1
        return count


class ETHOSBinary(Dataset):
    """Loads the ETHOS binary dataset."""

    labels_str_to_num = {
        'Hate': 1,
        'NotHate': 0
    }
    labels_num_to_str = {v: k for k, v in labels_str_to_num.items()}

    def load(self, split: Optional[str] = None) -> None:
        assert split in ['train', 'dev', 'test', None]
        with open(self.path) as fin:
            for line in fin:
                item = json.loads(line)
                if split:
                    if split == item['split']:
                        label_str = 'Hate' if int(round(float(item['label']))) == 1 else 'NotHate'
                        item['label'] = label_str
                        self._items.append(item)
                else:
                    label_str = 'Hate' if int(round(float(item['label']))) == 1 else 'NotHate'
                    item['label'] = label_str
                    self._items.append(item)

    def get_num_items(self) -> int:
        return len(self._items)

    def _add_items(self, item: Dict[str, Any]) -> None:
        label_str = 'Hate' if int(round(float(item['label']))) == 1 else 'NotHate'
        item['label'] = label_str
        self._items.append(item)
