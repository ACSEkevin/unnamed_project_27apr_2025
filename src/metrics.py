import motmetrics as mm
import pandas as pd

from torch import Tensor
from pandas import DataFrame


class MOTMetricWrapper:
    def __init__(self, matching_thres: float = .5, metrics: list[str] = None) -> None:
        self.matching_thres = matching_thres
        self.metrics = metrics if metrics else [
            "num_frames",
            # "hota_alpha", "assa_alpha", "deta_alpha", 
            "idf1", "mota", "motp", "num_switches"
        ]
        self.metric_names = [
            "NFrame",
            # f"HOTA@{matching_thres}", f"AssA@{matching_thres}", f"DetA@{matching_thres}",
            "IDF1", "MOTA", "MOTP", "IDs"
        ]
        self._name_map = {k: v for k, v in zip(self.metrics, self.metric_names)}

        self._acc = mm.MOTAccumulator(auto_id=True)
        self._met = mm.metrics.create()
        self._results = []
    
    def update(self, target_ids: Tensor, hypothesis_ids: Tensor, cost_matrix: Tensor):
        self._acc.update(
            target_ids.cpu().tolist(),
            hypothesis_ids.cpu().tolist(),
            cost_matrix.cpu().numpy()
        )

    def accumulate(self) -> DataFrame:
        """
        Accumulate partial result or result of events predicted from a single video.
        This is used for caculating final scores which reduces partial results.
        """
        summary = self.partial_result(render=False)
        self._results.append(summary)
        self._acc.reset()

        return summary
    
    def render_summary(self, summary: DataFrame):
        return mm.io.render_summary(
                summary, namemap=self._name_map
            )

    def partial_result(self, render: bool = True):
        """
        Return metric scores computed by data from accumulator.
        """
        summary: DataFrame = self._met.compute(
            self._acc, metrics=self.metrics, name="scores", return_dataframe=True
        )

        if render:
            rendered_summary = self.render_summary(summary)
            print(rendered_summary)

        return summary
    
    def result(self, std_out: bool = True):
        """
        Reduce partial results.
        """
        summary = pd.concat(self._results)
        data = summary.to_numpy() # [N, N_metrics]
        num_data = data.shape[0]
        reduced = data.sum(0, keepdims=True)
        reduced[:, 1:4] = reduced[:, 1:4] / num_data

        summary = DataFrame(reduced.tolist(), columns=self.metric_names)

        if std_out:
            print(summary)

        return summary

    def reset(self):
        self._acc.reset()
        self._results.clear()

    @property
    def events_(self):
        return self._acc.events
    
    @property
    def mot_events_(self):
        return self._acc.mot_events
    
    @property
    def accumulated_scores_(self):
        return self._results
    

__all__ = [
    "MOTMetricWrapper"
]