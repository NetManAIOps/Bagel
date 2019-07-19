from typing import Sequence, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import collections


def x_y_curve(data: Sequence[Dict], title: str=None, xlabel: str=None, ylabel: str=None, fig_params=None):
    if fig_params is None:
        fig_params = {}
    fig = plt.figure(**fig_params)
    handles = []
    for series in data:
        x = series.pop("x")
        y = series.pop("y")
        handle, = plt.plot(x, y, **series)
        if "label" in series:
            handles.append(handle)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=handles)
    return fig


def kpi_curve(data: Sequence[Dict], title: str=None, xlabel:str = None, ylabel: str=None, fig_params=None, ylim=None):
    import numpy as np
    from kpi_anomaly_detection.kpi_series import KPISeries
    from datetime import datetime
    if fig_params is None:
        fig_params = {}
    fig, ax = plt.subplots(**fig_params)
    handles = []
    for series in data:
        name = series.pop("name", None)  # because KPI use the name "label" for anomaly points
        std = series.pop("std", None)
        kpi_series = KPISeries(
            value=series.pop("value", series.pop("y")),
            timestamp=series.pop("timestamp", series.pop("x")),
            label=series.pop("label", None), missing=series.pop("missing", None), name=name)
        handle, = plt.plot([datetime.fromtimestamp(_) for _ in kpi_series.timestamp], kpi_series.value, label=name, **series)
        if name is not None:
            handles.append(handle)
        if std is not None:
            plt.fill_between([datetime.fromtimestamp(_) for _ in kpi_series.timestamp],
                             kpi_series.value - std, kpi_series.value + std,
                             alpha=series.get("fill_alpha", "0.5")
                             )

        # plot anomaly
        def _plot_anomaly(_series, _color):
            split_index = np.where(np.diff(_series) != 0)[0] + 1
            points = np.vstack((kpi_series.timestamp, kpi_series.value)).T.reshape(-1, 2)
            segments = np.split(points, split_index)
            for i in range(len(segments) - 1):
                segments[i] = np.concatenate([segments[i], [segments[i + 1][0]]])
            if _series[0] == 1:
                segments = segments[0::2]
            else:
                segments = segments[1::2]
            for line in segments:
                plt.plot([datetime.fromtimestamp(_) for _ in line[:, 0]], line[:, 1], _color)

        _plot_anomaly(kpi_series.label, "red")
        _plot_anomaly(kpi_series.missing, "orange")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.legend(handles=handles)
    return fig
