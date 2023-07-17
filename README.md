# Удаление тумана и дымки с оптических изображений

Этот проект представляет собой реализацию алгоритма темного канала для удаления тумана и дымки с оптических изображений.

## Описание алгоритма

### `dark_channel.py`

Алгоритм темного канала (dark channel) осуществляет удаление тумана и дымки из изображения путем следующих шагов:

1. Получение черно-белого канала изображения путем выбора минимального значения из цветных каналов (R, G, B).
2. Вычисление темного канала из черно-белого изображения с использованием скользящего окна заданного размера.
3. Определение атмосферного освещения на основе темного канала и исходного цветного изображения.
4. Восстановление сцены путем удаления тумана и дымки с исходного изображения с использованием полученных параметров.

### `filter.py`

Дополнительный файл `filter.py` содержит функции для улучшения результата алгоритма темного канала с помощью регуляризации:

1. Вычисление фильтра на основе окна заданного размера для улучшения результата алгоритма темного канала.
2. Улучшение результата алгоритма темного канала с использованием регуляризации.

![До:](image/until.jpg)
![После](image/before.png)
