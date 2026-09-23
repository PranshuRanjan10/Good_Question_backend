# Anomaly detector: precision / recall / F1 by label

Test month: April 2025 (23,767 five-minute windows, 1,021 anomalous). Trained on Nov-Mar; the split is by time, never random. Model decision threshold 0.8.

P = precision, R = recall. Per label, a method 'fires' when it reports that label (a rule named after it fires, or the model names it).

| Label | Windows | Rules P | R | F1 | Model P | R | F1 | Rules + model P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fatigue | 484 | 0.84 | 1.00 | 0.91 | 0.99 | 1.00 | 0.99 | 0.84 | 1.00 | 0.91 |
| excessive_idling | 135 | 0.34 | 0.58 | 0.43 | 0.42 | 0.23 | 0.30 | 0.34 | 0.59 | 0.44 |
| after_hours_use | 65 | 0.55 | 0.68 | 0.61 | 0.97 | 0.91 | 0.94 | 0.62 | 0.91 | 0.74 |
| overspeed | 40 | 0.67 | 1.00 | 0.80 | 0.76 | 0.93 | 0.83 | 0.67 | 1.00 | 0.80 |
| bucket_raised_travel | 38 | 0.72 | 1.00 | 0.83 | 0.95 | 0.92 | 0.93 | 0.72 | 1.00 | 0.83 |
| fast_swing | 37 | 0.95 | 0.95 | 0.95 | 1.00 | 0.97 | 0.99 | 0.95 | 1.00 | 0.97 |
| slope_exceeded | 33 | 0.97 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 0.98 |
| unauthorized_operator | 33 | 1.00 | 0.94 | 0.97 | 1.00 | 0.91 | 0.95 | 1.00 | 0.94 | 0.97 |
| harsh_operation | 32 | 0.97 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 0.98 |
| over_rev | 32 | 0.89 | 1.00 | 0.94 | 1.00 | 0.84 | 0.92 | 0.89 | 1.00 | 0.94 |
| seatbelt_off_while_moving* | 22 | 0.07 | 1.00 | 0.12 | 0.46 | 0.59 | 0.52 | 0.07 | 1.00 | 0.12 |
| overheating | 21 | 0.75 | 0.14 | 0.24 | 0.61 | 0.81 | 0.69 | 0.61 | 0.81 | 0.69 |
| low_productivity | 16 | n/a | n/a | n/a | 0.33 | 0.06 | 0.10 | 0.33 | 0.06 | 0.10 |
| operator_out_of_seat | 13 | 0.22 | 1.00 | 0.36 | 0.00 | 0.00 | 0.00 | 0.22 | 1.00 | 0.36 |
| fuel_theft | 11 | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 | 0.96 | 0.92 | 1.00 | 0.96 |
| overload | 5 | 0.71 | 1.00 | 0.83 | 1.00 | 0.80 | 0.89 | 0.71 | 1.00 | 0.83 |
| sensor_dropout | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| unsafe_refuelling | 2 | n/a | n/a | n/a | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Overall (any anomaly, binary):

| | Rules P | R | F1 | Model P | R | F1 | Rules + model P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Excluding seatbelt (target set) | 0.76 | 0.90 | 0.83 | 0.92 | 0.84 | 0.88 | 0.76 | 0.94 | 0.84 |
| All labels | 0.62 | 0.90 | 0.73 | 0.91 | 0.84 | 0.87 | 0.62 | 0.94 | 0.75 |

\* Known data limitation: the simulator only labels *injected* seatbelt episodes. Unbelted digging that happens naturally is labelled `normal` even though the alert is correct, so this label's precision is understated. It is reported on its own and left out of the overall target.

Rules use the same context flags the live row carries (declared break, engine started outside the schedule), rebuilt from `minute_telemetry.parquet` for evaluation.
Labels with no rule (`low_productivity`) are detected by the model only.
