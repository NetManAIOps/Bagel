from model import DonutX
import pandas as pd
import numpy as np
from kpi_series import KPISeries
from sklearn.metrics import precision_recall_curve
from evaluation_metric import range_lift_with_delay

df = pd.read_csv('sample_data.csv', header=0, index_col=None)
kpi = KPISeries(
    value = df.value,
    timestamp = df.timestamp,
    label = df.label,
    name = 'sample_data',
)

train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
train_kpi, train_kpi_mean, train_kpi_std = train_kpi.normalize(return_statistic=True)
valid_kpi = valid_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)
test_kpi = test_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)

model = DonutX(cuda=False, max_epoch=50, latent_dims=8, network_size=[100, 100])
model.fit(train_kpi.label_sampling(0.), valid_kpi)
y_prob = model.predict(test_kpi.label_sampling(0.))

y_prob = range_lift_with_delay(y_prob, test_kpi.label)
precisions, recalls, thresholds = precision_recall_curve(test_kpi.label, y_prob)
f1_scores = (2 * precisions * recalls) / (precisions + recalls)
print(f'best F1-score: {np.max(f1_scores[np.isfinite(f1_scores)])}')