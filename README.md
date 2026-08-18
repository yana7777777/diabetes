# Diabetes Analysis — анализ данных о диабете

Проект по анализу датасета **diabetes** из библиотеки `sklearn`.  
Цель — построить простую прогнозную модель на основе одного признака (`s6`) и сравнить с базовым прогнозом (среднее значение).

## Данные
- Датасет: `load_diabetes` из `sklearn.datasets`
- Признаки: 10 медицинских показателей
- Целевая переменная: `target` — прогрессия диабета через год

## Структура проекта
diabetes/
├── src/
│ └── diabetes_analysis.py # основной код
├── .gitignore
├── README.md
└── requirements.txt

## Как запустить
```bash
pip install -r requirements.txt
python src/diabetes_analysis.py