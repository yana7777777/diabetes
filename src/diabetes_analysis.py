import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.datasets import load_diabetes

print("Библиотеки загружены")

data = load_diabetes(as_frame=True)

# создайте DataFrame
df = data.frame.copy()

print("Размер таблицы:", df.shape)
display(df.head())

assert df.shape[0] == 442
assert df.shape[1] == 11
assert "target" in df.columns

import pandas as pd
from sklearn.datasets import load_diabetes

# 1. Загружаем датасет
diabetes = load_diabetes()

# 2. Создаем DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# 3. Сохраняем в CSV-файл на рабочий стол (или в любую папку)
df.to_csv('diabetes_dataset.csv', index=False)

print("Файл сохранен как 'diabetes_dataset.csv'")

# посчитайте min, max, mean, median для target
target_stats = {
    "min": df["target"].min(),
    "max": df["target"].max(),
    "mean": df["target"].mean(),
    "median": df["target"].median(),
}

stats_df = pd.DataFrame([target_stats]).T
stats_df.columns = ["value"]
display(stats_df.round(2))

assert target_stats["min"] < target_stats["mean"] < target_stats["max"]
assert target_stats["median"] > 0

plt.figure(figsize=(8, 4))
plt.hist(df["target"], bins=20)
plt.title("Распределение target")
plt.xlabel("target")
plt.ylabel("Количество")
plt.grid(True)
plt.tight_layout()
plt.show()


# baseline_prediction = среднее target
baseline_prediction = df["target"].mean()
df["baseline_pred"] = baseline_prediction

# создайте колонку baseline_pred
print("Baseline prediction:", round(baseline_prediction, 2))
display(df[["target", "baseline_pred"]].head())

assert baseline_prediction > 0
assert (df["baseline_pred"] == baseline_prediction).all()

# посчитайте абсолютную ошибку baseline
df["baseline_abs_error"] = (df["target"] - df["baseline_pred"]).abs()

# MAE = среднее абсолютной ошибки
baseline_mae = df["baseline_abs_error"].mean()

print("Baseline MAE:", round(baseline_mae, 2))
display(df[["target", "baseline_pred", "baseline_abs_error"]].head())

assert baseline_mae > 0
assert df["baseline_abs_error"].min() >= 0


feature = "s6"

plt.figure(figsize=(8, 5))

# постройте scatter plot s6 и target
plt.scatter(df[feature], df["target"], alpha=0.7)

plt.title("Связь s6 и target")
plt.xlabel(feature)
plt.ylabel("target")
plt.grid(True)
plt.tight_layout()
plt.show()

# посчитайте корреляцию bmi с target
correlation = df[feature].corr(df["target"])
print("Correlation s6-target:", round(correlation, 3))

assert abs(correlation) > 0.3



s6_mean = df[feature].mean()

# создайте группу high_s6 / low_s6
df["s6_group"] = df[feature].apply(lambda x: "high_s6" if x > s6_mean else "low_s6")

# средний target по группам
group_means = df.groupby("s6_group")["target"].mean()

# simple_pred по группе s6
df["simple_pred"] = df["s6_group"].map(group_means)

display(group_means.round(2))
display(df[[feature, "s6_group", "target", "simple_pred"]].head(10))

assert set(df["s6_group"].unique()) == {"high_s6", "low_s6"}
assert df["simple_pred"].notna().all()
assert group_means["high_s6"] > group_means["low_s6"]


# посчитайте ошибку simple_pred
df["simple_abs_error"] = (df["target"] - df["simple_pred"]).abs()

# посчитайте MAE простой модели
simple_mae = df["simple_abs_error"].mean()

comparison = pd.DataFrame([
    {"model": "baseline_mean", "MAE": baseline_mae},
    {"model": "simple_s6_groups", "MAE": simple_mae},
])

display(comparison.round(2))

improvement = baseline_mae - simple_mae
print("Улучшение MAE:", round(improvement, 2))

assert simple_mae > 0
assert simple_mae < baseline_mae

report = pd.DataFrame([
    {"metric": "rows", "value": len(df)},
    {"metric": "baseline_mae", "value": round(baseline_mae, 2)},
    {"metric": "simple_s6_mae", "value": round(simple_mae, 2)},
    {"metric": "mae_improvement", "value": round(improvement, 2)},
    {"metric": "s6_target_corr", "value": round(correlation, 3)},
])

report_path = "block03_simple_diabetes_report_s6.csv"  # ← можно поменять имя файла

# сохраните отчёт в CSV
report.to_csv(report_path, index=False)

display(report)
print("Файл сохранён:", report_path)

print("\nВывод:")
print("Мы сделали простой числовой прогноз по одному признаку (уровень сахара крови s6).")
print("Это учебная математика и информатика, а не медицинская рекомендация.")

assert report.shape[0] == 5