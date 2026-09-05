# Reverse audit: exact private mask replay

Дата запуска: 2026-09-05.

## Результат

В private восстановлено точное правило формирования synthetic gaps:

1. eligible-популяция — строки, где исходный target был доступен:
   `primary_ndvi.notna() OR is_synthetic_gap`;
2. размер eligible-популяции: **20753**;
3. скрывается `int(0.15 * N) = 3112` строк;
4. выбираются ordinal-индексы через `np.random.default_rng(43).choice(N, 3112,
   replace=False)`.

Replay совпал с каждым из **3112 / 3112**
флагов (`mismatch_rows = 0`). Это reverse-engineering маски, а не восстановление
самих target-значений: скрытые динамические поля остаются пустыми.

## Проверка уникальности seed

Сканирование seed `0..200` выполнено для той же eligible-популяции.
Лучшие строки:

```text
 seed  overlap  mismatch  exact
   43     3112         0   True
   63      516      5192  False
   27      512      5200  False
   74      511      5202  False
   95      507      5210  False
  101      506      5212  False
   92      505      5214  False
   59      501      5222  False
```

Seed 43 даёт точное совпадение; альтернативы дают лишь случайное частичное
пересечение. Повторяемость подтверждает, что порядок CSV и ordinal-популяция
сохранены.

## Что это даёт модели

Знание seed не раскрывает значения `primary_ndvi`, потому что target и все
сенсорные/погодные поля в gap-строках замаскированы. Практическая польза —
строить **точные synthetic folds** на известных target-строках и отдельно
проверять гипотезы для private-паттерна без подгонки по приблизительной маске.
Отдельного submission-кандидата только на replay маски не создавалось.

## Артефакты

- `reverse_mask_replay.py` — воспроизводимый replay;
- `reverse_mask_indices.csv` — строковые индексы, eligible-ordinal и проверка;
- `reverse_mask_seed_scan.csv` — сканирование seed;
- `reverse_mask_by_year.csv` — контрольные количества;
- `reverse_mask_summary.json` — машиночитаемая сводка.

Входной private SHA256: `3c5c0e27eef8266bcf6dce09c9b556c073cee3902c065a94e4ea7a59edb00993`. Файлы `outputs/model_dani_tuned*`
не читаются для построения replay и не изменялись.
