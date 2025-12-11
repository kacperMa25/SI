# pyright: standard
import math
from collections import defaultdict
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import KBinsDiscretizer


class BayesClassifier:
    def __init__(self) -> None:
        self._condProbs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self._aprioriProbs = defaultdict(int)

        self._n = 1
        self._df = DataFrame()
        self._widths = dict()
        self._mins = dict()
        self._use_log = False

    def getBin(self, col: str, value: float) -> int:
        return int(min(self._n - 1, (value - self._mins[col]) / self._widths[col]))

    def fit(
        self, dataFrame: DataFrame, n: int, laplace: bool = False, use_log: bool = False
    ):
        self._df = dataFrame.copy()
        self._n = n
        self._use_log = use_log
        self._condProbs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self._aprioriProbs = defaultdict(int)

        for col in self._df.columns:
            if col == "class":
                for classVal in self._df[col]:
                    self._aprioriProbs[classVal] += 1
                continue

            xMax = self._df[col].max()
            self._mins[col] = self._df[col].min()
            self._widths[col] = (xMax - self._mins[col]) / n
            if self._widths[col] == 0:
                self._widths[col] = 1

            self._df[col] = self._df[col].apply(lambda v: self.getBin(col, v))

        for _, row in self._df.iterrows():
            cls = row["class"]
            for col in self._df.columns:
                if col == "class":
                    continue
                val = row[col]
                self._condProbs[cls][col][val] += 1

        n_values = 0
        one = 0
        if laplace:
            n_values = n
            one = 1

        for cls in self._condProbs:
            for col in self._condProbs[cls]:
                total = sum(self._condProbs[cls][col].values())

                for val in self._condProbs[cls][col]:
                    self._condProbs[cls][col][val] = (
                        self._condProbs[cls][col][val] + one
                    ) / (total + n_values)

    def predict(self, sample: dict) -> int:
        temp = {}
        for cls in self._aprioriProbs:
            if self._use_log:
                score = math.log(self._aprioriProbs[cls])
                for x, val in sample.items():
                    bin_val = self.getBin(x, val)
                    prob = self._condProbs[cls][x][bin_val]
                    if prob > 0:
                        score += math.log(prob)
                    else:
                        score += math.log(1e-10)
            else:
                score = self._aprioriProbs[cls]
                for x, val in sample.items():
                    bin_val = self.getBin(x, val)
                    score *= self._condProbs[cls][x][bin_val]

            temp[cls] = score

        return max(temp, key=temp.get)

    def predict_probability(self, sample: dict, clsP) -> float:
        total = 0
        likeP = 0
        for cls in self._aprioriProbs:
            score = self._aprioriProbs[cls]
            for x, val in sample.items():
                bin_val = self.getBin(x, val)
                score *= self._condProbs[cls][x][bin_val]
            if clsP == cls:
                likeP = score
            total += score

        if total == 0:
            return 0.0
        return likeP / total


def calculate_accuracy(classifier, df: DataFrame) -> tuple:
    """Oblicza dokładność i statystyki dla każdej klasy"""
    correct = 0
    total = len(df)
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for _, row in df.iterrows():
        sample = row.drop("class").to_dict()
        predicted = classifier.predict(sample)
        actual = row["class"]

        class_stats[actual]["total"] += 1
        if predicted == actual:
            correct += 1
            class_stats[actual]["correct"] += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, dict(class_stats)


def run_basic_experiment(
    train_df: DataFrame,
    test_df: DataFrame,
    n: int = 5,
    laplace: bool = False,
    use_log: bool = False,
) -> tuple:
    """Uruchamia podstawowy eksperyment i zwraca wyniki"""
    classifier = BayesClassifier()
    classifier.fit(train_df.copy(), n=n, laplace=laplace, use_log=use_log)

    train_acc, train_stats = calculate_accuracy(classifier, train_df)
    test_acc, test_stats = calculate_accuracy(classifier, test_df)

    return classifier, train_acc, test_acc, train_stats, test_stats


def save_detailed_results(
    classifier: BayesClassifier, test_df: DataFrame, filename: str
):
    """Zapisuje szczegółowe wyniki do pliku CSV"""
    results = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        sample = row.drop("class").to_dict()
        predicted = classifier.predict(sample)
        actual = row["class"]
        is_correct = predicted == actual

        prob_predicted = classifier.predict_probability(sample, predicted)
        prob_actual = classifier.predict_probability(sample, actual)

        results.append(
            {
                "numer_probki": idx + 1,
                "etykieta_prawdziwa": actual,
                "etykieta_przewidziana": predicted,
                "poprawnosc": "TAK" if is_correct else "NIE",
                "prawdopodobienstwo_przewidzianej": round(prob_predicted, 6),
                "prawdopodobienstwo_prawdziwej": round(prob_actual, 6),
            }
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(filename, index=False, encoding="utf-8")
    print(f"Zapisano szczegółowe wyniki do {filename}")


def format_class_stats(stats: dict) -> str:
    """Formatuje statystyki klas do zapisu"""
    parts = []
    for cls in sorted(stats.keys()):
        parts.append(f"klasa_{cls}:{stats[cls]['correct']}/{stats[cls]['total']}")
    return "; ".join(parts)


def main():
    print("=" * 60)
    print("EKSPERYMENTY Z KLASYFIKATOREM NAIVE BAYES")
    print("=" * 60)

    df = read_csv("../seeds_dataset.csv")
    print("\nZbiór danych: seeds_dataset.csv")
    print(f"Liczba próbek: {len(df)}")
    print(f"Liczba atrybutów: {len(df.columns) - 1}")
    print(f"Klasy: {df['class'].unique()}")

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    experiments_results = []
    exp_id = 1

    # ============================================================
    # 1. EKSPERYMENT PODSTAWOWY
    # ============================================================
    print("\n" + "=" * 60)
    print("1. EKSPERYMENT PODSTAWOWY")
    print("=" * 60)

    classifier, train_acc, test_acc, train_stats, test_stats = run_basic_experiment(
        train_df.copy(), test_df.copy(), n=5, laplace=False, use_log=False
    )

    print(f"Dokładność na zbiorze uczącym: {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"Dokładność na zbiorze testowym:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Statystyki klas (test): {format_class_stats(test_stats)}")

    save_detailed_results(classifier, test_df, "test_basic.csv")

    experiments_results.append(
        {
            "id_eksperymentu": exp_id,
            "nazwa_zbioru": "seeds_dataset",
            "wariant_modelu": "dyskretny",
            "liczba_przedzialow": 5,
            "poprawka_laplace": "NIE",
            "uzycie_log": "NIE",
            "proporcje_train_test": "80/20",
            "dokladnosc_train": round(train_acc, 4),
            "dokladnosc_test": round(test_acc, 4),
            "statystyki_klas_test": format_class_stats(test_stats),
        }
    )
    exp_id += 1

    # ============================================================
    # 2. EKSPERYMENT Z LOGARYTMOWANIEM
    # ============================================================
    print("\n" + "=" * 60)
    print("2. EKSPERYMENT Z LOGARYTMOWANIEM (wzór 6.26)")
    print("=" * 60)

    _, train_acc_log, test_acc_log, train_stats_log, test_stats_log = (
        run_basic_experiment(
            train_df.copy(), test_df.copy(), n=5, laplace=False, use_log=True
        )
    )

    print(
        f"Dokładność na zbiorze uczącym: {train_acc_log:.4f} ({train_acc_log * 100:.2f}%)"
    )
    print(
        f"Dokładność na zbiorze testowym: {test_acc_log:.4f} ({test_acc_log * 100:.2f}%)"
    )
    print(f"Statystyki klas (test): {format_class_stats(test_stats_log)}")

    experiments_results.append(
        {
            "id_eksperymentu": exp_id,
            "nazwa_zbioru": "seeds_dataset",
            "wariant_modelu": "dyskretny",
            "liczba_przedzialow": 5,
            "poprawka_laplace": "NIE",
            "uzycie_log": "TAK",
            "proporcje_train_test": "80/20",
            "dokladnosc_train": round(train_acc_log, 4),
            "dokladnosc_test": round(test_acc_log, 4),
            "statystyki_klas_test": format_class_stats(test_stats_log),
        }
    )
    exp_id += 1

    # ============================================================
    # 3. EKSPERYMENT Z POPRAWKĄ LAPLACE'A
    # ============================================================
    print("\n" + "=" * 60)
    print("3. EKSPERYMENT Z POPRAWKĄ LAPLACE'A")
    print("=" * 60)

    _, train_acc_lap, test_acc_lap, train_stats_lap, test_stats_lap = (
        run_basic_experiment(
            train_df.copy(), test_df.copy(), n=5, laplace=True, use_log=False
        )
    )

    print(
        f"Dokładność na zbiorze uczącym: {train_acc_lap:.4f} ({train_acc_lap * 100:.2f}%)"
    )
    print(
        f"Dokładność na zbiorze testowym: {test_acc_lap:.4f} ({test_acc_lap * 100:.2f}%)"
    )
    print(f"Statystyki klas (test): {format_class_stats(test_stats_lap)}")
    print("\nPorównanie z wersją bez poprawki:")
    print(f"  Różnica dokładności (test): {(test_acc_lap - test_acc) * 100:+.2f}%")

    experiments_results.append(
        {
            "id_eksperymentu": exp_id,
            "nazwa_zbioru": "seeds_dataset",
            "wariant_modelu": "dyskretny",
            "liczba_przedzialow": 5,
            "poprawka_laplace": "TAK",
            "uzycie_log": "NIE",
            "proporcje_train_test": "80/20",
            "dokladnosc_train": round(train_acc_lap, 4),
            "dokladnosc_test": round(test_acc_lap, 4),
            "statystyki_klas_test": format_class_stats(test_stats_lap),
        }
    )
    exp_id += 1

    # ============================================================
    # 4. EKSPERYMENT ZALEŻNOŚCI OD LICZBY PRZEDZIAŁÓW
    # ============================================================
    print("\n" + "=" * 60)
    print("4. EKSPERYMENT ZALEŻNOŚCI OD LICZBY PRZEDZIAŁÓW")
    print("=" * 60)

    bin_values = [2, 3, 4, 5, 7, 10, 15, 20]

    for n_bins in bin_values:
        _, train_acc_n, test_acc_n, train_stats_n, test_stats_n = run_basic_experiment(
            train_df.copy(), test_df.copy(), n=n_bins, laplace=False, use_log=False
        )

        print(
            f"n={n_bins: 2d}: train={train_acc_n:.4f}, test={test_acc_n:.4f} | {format_class_stats(test_stats_n)}"
        )

        experiments_results.append(
            {
                "id_eksperymentu": exp_id,
                "nazwa_zbioru": "seeds_dataset",
                "wariant_modelu": "dyskretny",
                "liczba_przedzialow": n_bins,
                "poprawka_laplace": "NIE",
                "uzycie_log": "NIE",
                "proporcje_train_test": "80/20",
                "dokladnosc_train": round(train_acc_n, 4),
                "dokladnosc_test": round(test_acc_n, 4),
                "statystyki_klas_test": format_class_stats(test_stats_n),
            }
        )
        exp_id += 1

    # ============================================================
    # 5. EKSPERYMENT ZALEŻNOŚCI OD PROPORCJI PODZIAŁU
    # ============================================================
    print("\n" + "=" * 60)
    print("5. EKSPERYMENT ZALEŻNOŚCI OD PROPORCJI PODZIAŁU")
    print("=" * 60)

    split_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]

    for ratio in split_ratios:
        train_split = df.sample(frac=ratio, random_state=42)
        test_split = df.drop(train_split.index)

        _, train_acc_s, test_acc_s, train_stats_s, test_stats_s = run_basic_experiment(
            train_split.copy(), test_split.copy(), n=5, laplace=False, use_log=False
        )

        ratio_str = f"{int(ratio * 100)}/{int((1 - ratio) * 100)}"
        print(
            f"{ratio_str}: train={train_acc_s:.4f}, test={test_acc_s:.4f} | {format_class_stats(test_stats_s)}"
        )

        experiments_results.append(
            {
                "id_eksperymentu": exp_id,
                "nazwa_zbioru": "seeds_dataset",
                "wariant_modelu": "dyskretny",
                "liczba_przedzialow": 5,
                "poprawka_laplace": "NIE",
                "uzycie_log": "NIE",
                "proporcje_train_test": ratio_str,
                "dokladnosc_train": round(train_acc_s, 4),
                "dokladnosc_test": round(test_acc_s, 4),
                "statystyki_klas_test": format_class_stats(test_stats_s),
            }
        )
        exp_id += 1

    # ============================================================
    # 6. EKSPERYMENT Z SCIKIT-LEARN
    # ============================================================
    print("\n" + "=" * 60)
    print("6. EKSPERYMENT Z SCIKIT-LEARN")
    print("=" * 60)

    X_train = train_df.drop("class", axis=1).values
    y_train = train_df["class"].values
    X_test = test_df.drop("class", axis=1).values
    y_test = test_df["class"].values

    discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)

    cat_nb = CategoricalNB()
    cat_nb.fit(X_train_disc, y_train)

    train_pred_cat = cat_nb.predict(X_train_disc)
    test_pred_cat = cat_nb.predict(X_test_disc)

    train_acc_cat = np.mean(train_pred_cat == y_train)
    test_acc_cat = np.mean(test_pred_cat == y_test)

    cat_test_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, actual in zip(test_pred_cat, y_test):
        cat_test_stats[actual]["total"] += 1
        if pred == actual:
            cat_test_stats[actual]["correct"] += 1

    print("CategoricalNB (scikit-learn):")
    print(f"  Dokładność train: {train_acc_cat:.4f}")
    print(f"  Dokładność test:   {test_acc_cat:.4f}")
    print(f"  Statystyki klas:  {format_class_stats(dict(cat_test_stats))}")
    print("\nPorównanie z własną implementacją:")
    print(f"  Różnica dokładności (test): {(test_acc_cat - test_acc) * 100:+.2f}%")

    experiments_results.append(
        {
            "id_eksperymentu": exp_id,
            "nazwa_zbioru": "seeds_dataset",
            "wariant_modelu": "dyskretny (sklearn CategoricalNB)",
            "liczba_przedzialow": 5,
            "poprawka_laplace": "TAK (domyślnie w sklearn)",
            "uzycie_log": "TAK (wewnętrznie)",
            "proporcje_train_test": "80/20",
            "dokladnosc_train": round(train_acc_cat, 4),
            "dokladnosc_test": round(test_acc_cat, 4),
            "statystyki_klas_test": format_class_stats(dict(cat_test_stats)),
        }
    )
    exp_id += 1

    gauss_nb = GaussianNB()
    gauss_nb.fit(X_train, y_train)

    train_pred_gauss = gauss_nb.predict(X_train)
    test_pred_gauss = gauss_nb.predict(X_test)

    train_acc_gauss = np.mean(train_pred_gauss == y_train)
    test_acc_gauss = np.mean(test_pred_gauss == y_test)

    gauss_test_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, actual in zip(test_pred_gauss, y_test):
        gauss_test_stats[actual]["total"] += 1
        if pred == actual:
            gauss_test_stats[actual]["correct"] += 1

    print("\nGaussianNB (scikit-learn):")
    print(f"  Dokładność train: {train_acc_gauss:.4f}")
    print(f"  Dokładność test:  {test_acc_gauss:.4f}")
    print(f"  Statystyki klas: {format_class_stats(dict(gauss_test_stats))}")

    experiments_results.append(
        {
            "id_eksperymentu": exp_id,
            "nazwa_zbioru": "seeds_dataset",
            "wariant_modelu": "ciągły (sklearn GaussianNB)",
            "liczba_przedzialow": "N/A",
            "poprawka_laplace": "N/A",
            "uzycie_log": "TAK (wewnętrznie)",
            "proporcje_train_test": "80/20",
            "dokladnosc_train": round(train_acc_gauss, 4),
            "dokladnosc_test": round(test_acc_gauss, 4),
            "statystyki_klas_test": format_class_stats(dict(gauss_test_stats)),
        }
    )
    exp_id += 1

    # ============================================================
    # 7. EKSPERYMENT NA INNYM ZBIORZE UCI
    # ============================================================
    print("\n" + "=" * 60)
    print("7. EKSPERYMENT NA INNYM ZBIORZE UCI")
    print("=" * 60)

    from sklearn.datasets import load_iris

    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["class"] = iris.target

    print("\nZbiór danych:  Iris (UCI)")
    print(f"Liczba próbek:  {len(iris_df)}")
    print(f"Liczba atrybutów: {len(iris_df.columns) - 1}")
    print(f"Atrybuty: {list(iris.feature_names)}")
    print(f"Klasy: {list(iris.target_names)} (0, 1, 2)")

    iris_train = iris_df.sample(frac=0.8, random_state=42)
    iris_test = iris_df.drop(iris_train.index)

    _, iris_train_acc, iris_test_acc, iris_train_stats, iris_test_stats = (
        run_basic_experiment(
            iris_train.copy(), iris_test.copy(), n=5, laplace=False, use_log=False
        )
    )

    print("\nWyniki na zbiorze Iris:")
    print(f"  Dokładność train: {iris_train_acc:.4f}")
    print(f"  Dokładność test:  {iris_test_acc:.4f}")
    print(f"  Statystyki klas: {format_class_stats(iris_test_stats)}")

    experiments_results.append(
        {
            "id_eksperymentu": exp_id,
            "nazwa_zbioru": "iris (UCI)",
            "wariant_modelu": "dyskretny",
            "liczba_przedzialow": 5,
            "poprawka_laplace": "NIE",
            "uzycie_log": "NIE",
            "proporcje_train_test": "80/20",
            "dokladnosc_train": round(iris_train_acc, 4),
            "dokladnosc_test": round(iris_test_acc, 4),
            "statystyki_klas_test": format_class_stats(iris_test_stats),
        }
    )
    exp_id += 1

    _, iris_train_acc_lap, iris_test_acc_lap, _, iris_test_stats_lap = (
        run_basic_experiment(
            iris_train.copy(), iris_test.copy(), n=5, laplace=True, use_log=False
        )
    )

    print("\nWyniki z poprawką Laplace'a:")
    print(f"  Dokładność train: {iris_train_acc_lap:.4f}")
    print(f"  Dokładność test:  {iris_test_acc_lap:.4f}")

    experiments_results.append(
        {
            "id_eksperymentu": exp_id,
            "nazwa_zbioru": "iris (UCI)",
            "wariant_modelu": "dyskretny",
            "liczba_przedzialow": 5,
            "poprawka_laplace": "TAK",
            "uzycie_log": "NIE",
            "proporcje_train_test": "80/20",
            "dokladnosc_train": round(iris_train_acc_lap, 4),
            "dokladnosc_test": round(iris_test_acc_lap, 4),
            "statystyki_klas_test": format_class_stats(iris_test_stats_lap),
        }
    )
    exp_id += 1

    # ============================================================
    # ZAPIS WYNIKÓW DO PLIKU EKSPERYMENTY.CSV
    # ============================================================
    print("\n" + "=" * 60)
    print("ZAPIS WYNIKÓW")
    print("=" * 60)

    df_experiments = pd.DataFrame(experiments_results)
    df_experiments.to_csv("eksperymenty.csv", index=False, encoding="utf-8")
    print("Zapisano podsumowanie eksperymentów do eksperymenty.csv")
    print(f"Łączna liczba eksperymentów: {len(experiments_results)}")

    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(df_experiments.to_string(index=False))


if __name__ == "__main__":
    main()
