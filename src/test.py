"""
Exploratory Data Analysis for Credit Default Dataset.

This script performs EDA on credit default data and generates visualizations
for the analysis report.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path: str = "Credit_Default.csv") -> pd.DataFrame:
    """
    Load the credit default dataset and perform initial preprocessing.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded and preprocessed dataset
    """
    df = pd.read_csv(file_path)
    # Round all numeric columns to 2 decimal places
    numeric_cols = df.select_dtypes(include=["float64"]).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    return df


def analyze_missing_values(df: pd.DataFrame) -> dict:
    """
    Analyze missing values in the dataset.

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        dict: Dictionary containing missing values analysis
    """
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100

    # Create missing values plot
    if missing_values.sum() > 0:
        plt.figure(figsize=(10, 6))
        missing_percentages.plot(kind="bar")
        plt.title("Missing Values by Column")
        plt.ylabel("Percentage of Missing Values")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("./for_pages/png/missing_values.png")
        plt.close()

    return {
        "missing_counts": missing_values.to_dict(),
        "missing_percentages": missing_percentages.to_dict(),
    }


def create_pairplot(df: pd.DataFrame) -> None:
    """
    Create pairwise distribution plots for numerical features.

    Args:
        df (pd.DataFrame): Input dataset
    """
    # Automatically determine numerical columns (excluding the target variable)
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    numerical_cols = numerical_cols[numerical_cols != "Default"].tolist()

    sns.pairplot(df, vars=numerical_cols, hue="Default", diag_kind="hist")
    plt.savefig("./for_pages/png/pairplot.png")
    plt.close()


def analyze_correlations(df: pd.DataFrame) -> None:
    """
    Analyze and visualize correlations between numerical features.

    Args:
        df (pd.DataFrame): Input dataset
    """
    # Include all numeric columns including Default
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    corr_matrix = df[numerical_cols].corr().round(2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("./for_pages/png/correlation_matrix.png")
    plt.close()


def analyze_class_balance(df: pd.DataFrame) -> dict:
    """
    Analyze the balance of default/non-default classes.

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        dict: Dictionary containing class balance statistics
    """
    class_counts = df["Default"].value_counts()
    class_percentages = ((class_counts / len(df)) * 100).round(2)

    plt.figure(figsize=(8, 6))
    class_counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Default Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("./for_pages/png/class_balance.png")
    plt.close()

    return {
        "counts": class_counts.to_dict(),
        "percentages": class_percentages.to_dict(),
    }


def main():
    """Main function to run the EDA pipeline."""
    # Load data
    df = load_data()

    # Print basic information
    print("\nDataset Overview:")
    print("-" * 50)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nColumn names:", list(df.columns))
    print("\nData types:\n", df.dtypes, sep="")

    # Analyze missing values
    missing_analysis = analyze_missing_values(df)
    print("\nMissing Values Analysis:")
    print("-" * 50)
    print("Missing value counts:", missing_analysis["missing_counts"])
    print("Missing value percentages:", missing_analysis["missing_percentages"])

    # Create visualizations
    create_pairplot(df)
    analyze_correlations(df)
    class_balance = analyze_class_balance(df)

    # Print class balance information
    print("\nClass Balance Analysis:")
    print("-" * 50)
    print("Class counts:", class_balance["counts"])
    print("Class percentages:", class_balance["percentages"])
    print("-" * 50)

    # Generate Quarto report
    generate_quarto_report(
        df,
        missing_analysis,
        class_balance,
    )


def generate_quarto_report(
    df: pd.DataFrame,
    missing_analysis: dict,
    class_balance: dict,
) -> None:
    """
    Generate a Quarto report with the EDA results.

    Args:
        missing_analysis (dict): Missing values analysis results
        class_balance (dict): Class balance analysis results
        descriptive_stats (dict): Descriptive statistics of the dataset
    """
    if sum(missing_analysis["missing_counts"].values()) > 0:
        result = "![Распределение пропущенных значений](./png/missing_values.png)"  # noqa: E501
    else:
        result = "В наборе данных не было обнаружено пропущенных значений."

    report_content = f"""---
title: "Отчет об анализе дефолта по кредитам"
---

## Предварительный обзор данных

В этом отчете представлены результаты разведочного анализа данных (EDA),
проведенного на датасете Credit Default.
В наборе данных содержится информация о клиентах,
включая их доход, возраст, сумму кредита и статус дефолта.

Несколько первых строк набора данных:
```python
{df.head()}
```

## Анализ пропущенных значений

```python
Число пропущенных значений: {missing_analysis['missing_counts']}
Процент пропущенных значений: {missing_analysis['missing_percentages']}
```

{result}

## Диаграммы попарного распределения признаков

![Попарные распределения признаков](./png/pairplot.png)

## Корреляционный анализ

Корреляционная матрица, показывающая взаимосвязь между числовыми признаками:

![Корреляционная матрица](./png/correlation_matrix.png)

## Анализ баланса классов

```python
Число представителей каждого класса: {class_balance['counts']}
Процентное соотношение классов: {class_balance['percentages']}
```

![Баланс классов](./png/class_balance.png)

## Заключение и выводы

#### 1. Распределение по классам.

В наборе данных наблюдается значительная несбалансированность классов.

#### 2. Корреляции.

Существуют заметные корреляции между некоторыми характеристиками:

    - суммой кредита и годовым доходом,

    - отношением суммы кредита к доходу клиента и суммой кредита.

#### 3. Распределения характеристик.

Распределение характеристик значительно отличается в случаях дефолта и
недефолта.

В датасете дефолт произошел исключительно для людей моложе 40 лет
для суммы кредита в основном превыщающей 5000.

Годовой доход не оказывает влияение на вероятность дефолта.
"""

    try:
        with open("./for_pages/index.qmd", "w", encoding="utf-8") as f:
            f.write(report_content)
        print("Файл 'index.qmd' успешно сохранен")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")


if __name__ == "__main__":
    main()
