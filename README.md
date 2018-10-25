# Linear regression test
Код этого проекта - попытка подобрать модель, описывающую линейную зависимость результативного признака y от факторных признаков x_1,x_2,x_3 и x_4. 

*yi = a1x_1i + a2x_2i + a3x_3i + a4x_4i + ei*, где е - вектор ошибки.

Минимизируя его получим искомые значения коэффициентов а1, а2, а3 и а4. Интересует суммарная ошибка по всему набору данных, но просто просуммировать компоненты будет ошибкой, т.к. их знак не фиксирован, и они могут взаимоуничтожиться. Значит, требуется рассмотреть сумму квадратов компонент вектора е. Прибегнув к методу наименьших квадратов, оказывается, что для поиска приближенного решения достаточно вычислить матрицу корреляции, стандартное отклонение и матожидание факторных случайных величин.

Python имеет встроенные инструенты для вычисления модели и прогнозирования.

# Running the tests

Входные данные : таблица значений всех параметров.

Выходные данные: статистика в формате таблицы model.summary()

Согласно построенной по данному датасету модели *yi = 2.3941x_1 -1.6865x_2 + 15.1768x_3 + 51.0001x_4*

По итогам анализа линейная модель объясняет лишь 60.2% дисперсии зависимых переменных.