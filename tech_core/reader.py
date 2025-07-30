import os
import io
import pandas as pd

class FastCSVChunkReader:
    def __init__(self, path, padding=0, batch_size=1000, has_header=True):
        self.path = path
        self.padding = padding
        self.batch_size = batch_size
        self.offsets = []
        self.has_header = has_header
        self._build_offsets()
        self.current_idx = 0
        self.asset_names = []
        self.end_idx = len(self.offsets) - 1

    def _build_offsets(self):
        with open(self.path, 'rb') as f:
            pos = f.tell()
            if self.has_header:
                f.readline()  # skip header
                pos = f.tell()
            self.offsets.append(pos)
            while line := f.readline():
                self.offsets.append(f.tell())

        self.total_rows = len(self.offsets)

    def __iter__(self):
        return self

    def set_split(self, start_row: int, end_row: int):
        self.current_idx = start_row
        self.end_idx = end_row

    def __next__(self):

        if self.current_idx >= self.end_idx:
            raise StopIteration

        start = self.current_idx
        end = min(self.current_idx + self.batch_size + self.padding, self.end_idx)

        with open(self.path, 'rb') as f:
            f.seek(self.offsets[start])
            lines = f.read(self.offsets[end] - self.offsets[start]).decode('utf-8')
            if self.has_header:
                with open(self.path, 'r') as h:
                    header = h.readline()
                result = header + lines
            else:
                result = lines

        self.current_idx += self.batch_size
        tmp = pd.read_csv(io.StringIO(result)).set_index('Unnamed: 0').rename_axis(None)
        tmp.index = pd.to_datetime(tmp.index)
        self.asset_names = tmp.columns.tolist()
        return tmp

    def reset(self):
        self.current_idx = 0