from bisect import bisect_right
import pandas as pd

class SplitManager:
    def __init__(self, split_dates: list[pd.Timestamp], split_names: list[str]):
        assert len(split_dates) + 1 == len(split_names)
        self.split_dates = split_dates
        self.split_names = split_names

    def get_split_name(self, dt: pd.Timestamp) -> str:
        idx = bisect_right(self.split_dates, dt)
        return self.split_names[idx]

    def get_split_bounds(self, all_times: list[pd.Timestamp], name: str) -> tuple[int, int]:
        idx = self.split_names.index(name)
        start_time = self.split_dates[idx - 1] if idx > 0 else pd.Timestamp.min
        end_time = self.split_dates[idx] if idx < len(self.split_dates) else pd.Timestamp.max

        start_idx = next(i for i, t in enumerate(all_times) if t >= start_time)
        end_idx = next((i for i, t in enumerate(all_times) if t > end_time), len(all_times))

        return start_idx, end_idx
