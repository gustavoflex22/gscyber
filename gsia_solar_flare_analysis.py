from __future__ import annotations

import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as PdfImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "outputs"
FINAL_DIR = ROOT / "gsia"

DATA_FILES = [DATA_DIR / "flare.data1", DATA_DIR / "flare.data2"]
RANDOM_SEED = 42
GROUP_MEMBERS = [
    "Vinicius Issa Gois (RM553814)",
    "Vinicius Caetano (RM552904)",
    "Gustavo Manganeli Felex (RM554242)",
    "Gustavo Bonani (RM553493)",
    "Wesley Leopoldino (RM553496)",
]

FEATURE_COLUMNS = [
    "zurich_class",
    "largest_spot_size",
    "spot_distribution",
    "activity",
    "evolution",
    "previous_flare_activity",
    "historically_complex",
    "became_complex",
    "area",
    "largest_spot_area",
]

TARGET_COLUMNS = ["c_flares", "m_flares", "x_flares"]


@dataclass(frozen=True)
class Metrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    f2: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int


class CategoricalNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_log_prior: dict[int, float] = {}
        self.feature_log_prob: dict[int, dict[str, dict[str, float]]] = {}
        self.feature_values: dict[str, list[str]] = {}

    def fit(self, rows: pd.DataFrame, target: pd.Series) -> None:
        class_values = sorted(target.unique().tolist())
        total_rows = len(target)

        for column in FEATURE_COLUMNS:
            self.feature_values[column] = sorted(rows[column].astype(str).unique().tolist())

        for class_value in class_values:
            class_mask = target == class_value
            class_rows = rows.loc[class_mask, FEATURE_COLUMNS].astype(str)
            class_count = int(class_mask.sum())
            self.class_log_prior[int(class_value)] = math.log(class_count / total_rows)
            self.feature_log_prob[int(class_value)] = {}

            for column in FEATURE_COLUMNS:
                value_counts = class_rows[column].value_counts().to_dict()
                possible_values = self.feature_values[column]
                denominator = class_count + self.alpha * len(possible_values)
                self.feature_log_prob[int(class_value)][column] = {}

                for value in possible_values:
                    count = int(value_counts.get(value, 0))
                    probability = (count + self.alpha) / denominator
                    self.feature_log_prob[int(class_value)][column][value] = math.log(probability)

    def predict_proba_positive(self, rows: pd.DataFrame) -> np.ndarray:
        scores: list[float] = []

        for _, row in rows[FEATURE_COLUMNS].astype(str).iterrows():
            log_scores: dict[int, float] = {}
            for class_value, prior in self.class_log_prior.items():
                log_score = prior
                for column in FEATURE_COLUMNS:
                    value = row[column]
                    value_probs = self.feature_log_prob[class_value][column]
                    fallback = math.log(self.alpha / (self.alpha * len(value_probs) + 1))
                    log_score += value_probs.get(value, fallback)
                log_scores[class_value] = log_score

            max_log = max(log_scores.values())
            exp_scores = {key: math.exp(value - max_log) for key, value in log_scores.items()}
            denominator = sum(exp_scores.values())
            scores.append(exp_scores.get(1, 0.0) / denominator)

        return np.array(scores)


def prepare_dirs() -> None:
    for path in [FIGURE_DIR, OUTPUT_DIR, FINAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_solar_flare_data() -> pd.DataFrame:
    rows: list[list[str]] = []

    for source_path in DATA_FILES:
        with source_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("*"):
                    continue
                rows.append(line.split())

    columns = FEATURE_COLUMNS + TARGET_COLUMNS
    frame = pd.DataFrame(rows, columns=columns)

    for column in FEATURE_COLUMNS[3:] + TARGET_COLUMNS:
        frame[column] = frame[column].astype(int)

    frame["any_c_flare"] = (frame["c_flares"] > 0).astype(int)
    frame["relevant_flare"] = ((frame["m_flares"] + frame["x_flares"]) > 0).astype(int)
    frame["total_flares"] = frame[TARGET_COLUMNS].sum(axis=1)
    frame["source_quality"] = ["data1"] * count_rows(DATA_FILES[0]) + ["data2"] * count_rows(DATA_FILES[1])
    return frame


def count_rows(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("*"):
                total += 1
    return total


def stratified_split(frame: pd.DataFrame, target_column: str, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(RANDOM_SEED)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in frame.groupby(target_column):
        indices = group.index.tolist()
        random.shuffle(indices)
        test_size = max(1, round(len(indices) * test_fraction))
        test_indices.extend(indices[:test_size])
        train_indices.extend(indices[test_size:])

    return frame.loc[train_indices].sample(frac=1, random_state=RANDOM_SEED), frame.loc[test_indices].sample(
        frac=1,
        random_state=RANDOM_SEED,
    )


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Metrics:
    y_pred = (y_score >= threshold).astype(int)
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (true_positive + true_negative) / len(y_true)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    f2 = (5 * precision * recall) / max((4 * precision) + recall, 1e-12)

    return Metrics(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        f2=f2,
        false_positives=false_positive,
        false_negatives=false_negative,
        true_positives=true_positive,
        true_negatives=true_negative,
    )


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Metrics:
    candidates = [round(value, 2) for value in np.linspace(0.05, 0.75, 71)]
    metrics = [compute_metrics(y_true, y_score, threshold) for threshold in candidates]
    return max(metrics, key=lambda item: (item.f2, item.recall, item.precision))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    color: tuple[int, int, int] = (15, 118, 110),
) -> None:
    width, height = 1200, 760
    margin_left, margin_bottom, margin_top, margin_right = 150, 120, 95, 50
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 30), title, fill=(15, 23, 42), font=font(34, bold=True))
    draw.text((30, 340), y_label, fill=(71, 85, 105), font=font(20))

    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1.0
    bar_gap = 28
    bar_width = max(20, int((chart_width - bar_gap * (len(values) - 1)) / max(len(values), 1)))

    draw.line((margin_left, margin_top, margin_left, margin_top + chart_height), fill=(148, 163, 184), width=2)
    draw.line(
        (margin_left, margin_top + chart_height, width - margin_right, margin_top + chart_height),
        fill=(148, 163, 184),
        width=2,
    )

    for index, (label, value) in enumerate(zip(labels, values)):
        left = margin_left + index * (bar_width + bar_gap)
        bar_height = int((value / max_value) * (chart_height - 24))
        top = margin_top + chart_height - bar_height
        draw.rounded_rectangle(
            (left, top, left + bar_width, margin_top + chart_height),
            radius=8,
            fill=color,
        )
        draw.text((left, top - 30), f"{value:.1f}", fill=(15, 23, 42), font=font(20, bold=True))
        draw.text((left, margin_top + chart_height + 16), label, fill=(51, 65, 85), font=font(18))

    canvas.save(path)


def draw_confusion_matrix(path: Path, metrics: Metrics) -> None:
    width, height = 1000, 760
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 40), "Matriz de confusão no teste", fill=(15, 23, 42), font=font(34, bold=True))
    draw.text((80, 88), "Classe positiva: ocorrência de flare M ou X nas próximas 24h", fill=(71, 85, 105), font=font(20))

    cells = [
        ("TN", metrics.true_negatives, "Previsto sem evento / Real sem evento", (204, 251, 241)),
        ("FP", metrics.false_positives, "Previsto evento / Real sem evento", (254, 226, 226)),
        ("FN", metrics.false_negatives, "Previsto sem evento / Real com evento", (255, 237, 213)),
        ("TP", metrics.true_positives, "Previsto evento / Real com evento", (219, 234, 254)),
    ]
    positions = [(110, 180), (520, 180), (110, 450), (520, 450)]

    for (label, value, subtitle, color), (x, y) in zip(cells, positions):
        draw.rounded_rectangle((x, y, x + 360, y + 210), radius=16, fill=color, outline=(148, 163, 184), width=2)
        draw.text((x + 24, y + 22), label, fill=(15, 23, 42), font=font(28, bold=True))
        draw.text((x + 24, y + 68), str(value), fill=(15, 23, 42), font=font(54, bold=True))
        draw.text((x + 24, y + 142), subtitle, fill=(51, 65, 85), font=font(17))

    canvas.save(path)


def create_figures(frame: pd.DataFrame, metrics: Metrics, risk_table: pd.DataFrame, feature_table: pd.DataFrame) -> None:
    target_counts = frame["relevant_flare"].value_counts().sort_index()
    draw_bar_chart(
        FIGURE_DIR / "01_distribuicao_alvo.png",
        "Distribuição do alvo: flare relevante é raro",
        ["Sem M/X", "Com M/X"],
        [float(target_counts.get(0, 0)), float(target_counts.get(1, 0))],
        "Regiões ativas",
    )

    draw_bar_chart(
        FIGURE_DIR / "02_risco_por_zurich.png",
        "Taxa de flares relevantes por classe Zurich",
        risk_table["zurich_class"].astype(str).tolist(),
        (risk_table["event_rate"] * 100).round(1).tolist(),
        "% com M/X",
        color=(3, 105, 161),
    )

    draw_confusion_matrix(FIGURE_DIR / "03_matriz_confusao.png", metrics)

    draw_bar_chart(
        FIGURE_DIR / "04_lift_variaveis.png",
        "Categorias com maior lift observado",
        feature_table["feature_value"].astype(str).tolist(),
        feature_table["lift"].round(2).tolist(),
        "Lift vs. média",
        color=(180, 83, 9),
    )


def build_report(summary: dict[str, float | int | str], metrics: Metrics) -> None:
    report_path = FINAL_DIR / "gsia_relatorio.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyPT", parent=styles["BodyText"], fontSize=10.5, leading=15))
    styles.add(ParagraphStyle(name="TitlePT", parent=styles["Title"], fontSize=22, leading=27, spaceAfter=14))
    styles.add(ParagraphStyle(name="HeadingPT", parent=styles["Heading2"], fontSize=15, leading=19, spaceBefore=10))

    document = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    story: list[object] = []

    def p(text: str, style_name: str = "BodyPT") -> None:
        story.append(Paragraph(text, styles[style_name]))
        story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("GSIA - Previsão de risco de flares solares com Machine Learning", styles["TitlePT"]))
    p("<b>Integrantes:</b> " + "; ".join(GROUP_MEMBERS))
    p(
        "<b>Resumo técnico.</b> A investigação avaliou se atributos de regiões ativas do Sol conseguem sinalizar "
        "risco de flares moderados ou severos (classes M ou X) nas próximas 24 horas. O evento é raro: "
        f"{summary['positive_rate']:.1f}% das {summary['rows']} observações tiveram flare M/X. "
        "Por isso, a solução priorizou recall e interpretação de risco, não apenas acurácia."
    )
    p(
        f"O modelo Naive Bayes categórico atingiu recall de {metrics.recall:.2f}, precisão de {metrics.precision:.2f} "
        f"e F2-score de {metrics.f2:.2f} no conjunto de teste, usando limiar {metrics.threshold:.2f}. "
        "Em uma aplicação de monitoramento espacial, esse ajuste é coerente porque deixar de alertar um evento "
        "relevante tende a ser mais grave do que acionar investigação adicional."
    )

    story.append(Paragraph("Flares relevantes são raros e criam um problema desbalanceado", styles["HeadingPT"]))
    p(
        "A maior parte das regiões ativas não gerou flares M ou X. Esse desbalanceamento torna a acurácia isolada "
        "pouco informativa: um modelo que sempre prevê ausência de evento já pareceria correto na maioria dos casos, "
        "mas seria inútil para priorização de risco."
    )
    story.append(PdfImage(str(FIGURE_DIR / "01_distribuicao_alvo.png"), width=16.5 * cm, height=10.4 * cm))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("A classe Zurich concentra diferenças úteis de risco", styles["HeadingPT"]))
    p(
        "A classe Zurich, que resume características morfológicas da região ativa, mostra diferenças claras na taxa "
        "observada de eventos M/X. Essas diferenças não provam causalidade, mas ajudam a construir uma fila de "
        "priorização para observação posterior."
    )
    story.append(PdfImage(str(FIGURE_DIR / "02_risco_por_zurich.png"), width=16.5 * cm, height=10.4 * cm))
    story.append(PageBreak())

    story.append(Paragraph("Modelo e avaliação favorecem detecção de eventos", styles["HeadingPT"]))
    p(
        "Foi treinado um Naive Bayes categórico com suavização de Laplace, adequado para variáveis discretas e "
        "fácil de explicar. O alvo binário foi definido como 1 quando a região produziu ao menos um flare M ou X "
        "nas 24 horas seguintes. O limiar foi escolhido por F2-score, que dá mais peso ao recall."
    )
    story.append(PdfImage(str(FIGURE_DIR / "03_matriz_confusao.png"), width=16.5 * cm, height=12.5 * cm))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Categorias com maior lift indicam sinais observacionais promissores", styles["HeadingPT"]))
    p(
        "O lift compara a taxa de evento dentro de uma categoria com a taxa média do dataset. Valores acima de 1 "
        "indicam categorias nas quais flares M/X apareceram proporcionalmente mais. Como algumas categorias têm "
        "amostra pequena, o resultado deve ser lido como hipótese analítica, não como regra operacional definitiva."
    )
    story.append(PdfImage(str(FIGURE_DIR / "04_lift_variaveis.png"), width=16.5 * cm, height=10.4 * cm))

    story.append(Paragraph("Escopo, dados e definições", styles["HeadingPT"]))
    p(
        "Dataset escolhido: UCI Machine Learning Repository - Solar Flare Dataset. Cada linha representa uma região "
        "ativa do Sol, descrita por dez atributos categóricos/ordinais, e três contagens-alvo: flares C, M e X "
        "produzidos nas 24 horas seguintes. Foram usadas as duas partições públicas, totalizando "
        f"{summary['rows']} observações."
    )

    story.append(Paragraph("Limitações e robustez", styles["HeadingPT"]))
    p(
        "O dataset é histórico, pequeno para padrões atuais e tem poucos eventos severos. A divisão treino/teste foi "
        "estratificada, mas as métricas podem variar com outra amostra. O modelo também assume independência "
        "condicional entre atributos, uma simplificação forte. Ainda assim, a análise é útil pedagogicamente porque "
        "expõe desbalanceamento, trade-off entre falso positivo e falso negativo e interpretação de variáveis."
    )

    story.append(Paragraph("Próximos passos recomendados", styles["HeadingPT"]))
    p(
        "Para evoluir o trabalho, o grupo poderia comparar Naive Bayes com árvore de decisão, regressão logística ou "
        "Random Forest, incluir validação cruzada e calibrar probabilidades. Em uma aplicação real, também seria "
        "necessário atualizar os dados com fontes solares modernas e definir o custo operacional de cada tipo de erro."
    )

    document.build(story)


def build_presentation(summary: dict[str, float | int | str], metrics: Metrics) -> None:
    deck_path = FINAL_DIR / "gsia_apresentacao.pdf"
    document = SimpleDocTemplate(
        str(deck_path),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SlideTitle", parent=styles["Title"], fontSize=28, leading=34, spaceAfter=16))
    styles.add(ParagraphStyle(name="SlideBody", parent=styles["BodyText"], fontSize=16, leading=22))
    story: list[object] = []

    slides = [
        (
            "GSIA - Flares solares",
            "Integrantes: "
            + "; ".join(GROUP_MEMBERS)
            + "\n\nPergunta: atributos de regiões ativas do Sol ajudam a priorizar risco de flares M ou X nas próximas 24 horas?",
            None,
        ),
        (
            "Problema desbalanceado",
            f"Apenas {summary['positive_rate']:.1f}% das {summary['rows']} observações tiveram flare relevante. "
            "Acurácia sozinha não basta para avaliar um sistema de alerta.",
            FIGURE_DIR / "01_distribuicao_alvo.png",
        ),
        (
            "Sinais de risco aparecem nas variáveis",
            "A classe Zurich e algumas categorias morfológicas concentram taxas maiores de flare M/X, sugerindo "
            "utilidade para triagem observacional.",
            FIGURE_DIR / "02_risco_por_zurich.png",
        ),
        (
            "Modelo interpretável",
            f"Naive Bayes categórico: recall {metrics.recall:.2f}, precisão {metrics.precision:.2f}, "
            f"F2 {metrics.f2:.2f}. O limiar favorece não perder eventos relevantes.",
            FIGURE_DIR / "03_matriz_confusao.png",
        ),
        (
            "Conclusão",
            "O dataset permite uma investigação coerente de IA/ML: há sinal preditivo, mas o baixo número de eventos "
            "severos exige cautela, validação adicional e atualização com dados solares modernos.",
            None,
        ),
    ]

    for index, (title, body, image_path) in enumerate(slides):
        if index > 0:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["SlideTitle"]))
        story.append(Paragraph(body, styles["SlideBody"]))
        story.append(Spacer(1, 0.5 * cm))
        if image_path is not None:
            story.append(PdfImage(str(image_path), width=19.5 * cm, height=12.4 * cm))

    document.build(story)


def write_supporting_files(
    frame: pd.DataFrame,
    risk_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    metrics: Metrics,
    summary: dict[str, float | int | str],
) -> None:
    frame.to_csv(FINAL_DIR / "gsia_dataset_tratado.csv", index=False)
    risk_table.to_csv(FINAL_DIR / "gsia_risco_por_classe_zurich.csv", index=False)
    feature_table.to_csv(FINAL_DIR / "gsia_lift_variaveis.csv", index=False)

    metrics_payload = {
        "threshold": metrics.threshold,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "specificity": metrics.specificity,
        "f1": metrics.f1,
        "f2": metrics.f2,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "true_positives": metrics.true_positives,
        "true_negatives": metrics.true_negatives,
        "summary": summary,
    }
    (FINAL_DIR / "gsia_metricas.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GSIA - Solar Flare Dataset\n",
                    "\n",
                    "Este notebook acompanha o script `gsia_solar_flare_analysis.py`. Execute o script para reproduzir dados tratados, métricas, gráficos, relatório e apresentação.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Reproduz toda a análise\n",
                    "%run gsia_solar_flare_analysis.py\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (FINAL_DIR / "gsia_notebook.ipynb").write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = """# GSIA - IA e Machine Learning com dados espaciais

Tema escolhido: previsão de risco de flares solares usando o Solar Flare Dataset da UCI.

## Integrantes

- Vinicius Issa Gois (RM553814)
- Vinicius Caetano (RM552904)
- Gustavo Manganeli Felex (RM554242)
- Gustavo Bonani (RM553493)
- Wesley Leopoldino (RM553496)

## Pergunta investigada

Atributos observacionais de regiões ativas do Sol ajudam a priorizar quais regiões têm maior risco de produzir flares moderados ou severos (M ou X) nas próximas 24 horas?

## Como reproduzir

```bash
python3 gsia_solar_flare_analysis.py
```

O script lê os arquivos em `data/`, trata o dataset, treina o modelo Naive Bayes categórico, calcula métricas, gera figuras e exporta relatório/apresentação.

## Arquivos principais

- `gsia_relatorio.pdf`: relatório explicativo.
- `gsia_apresentacao.pdf`: apresentação breve.
- `gsia_solar_flare_analysis.py`: código completo e comentado.
- `gsia_notebook.ipynb`: notebook simples para execução do script.
- `data/`: dados originais baixados da UCI.
- `figures/`: gráficos usados no relatório.
- `gsia_metricas.json`: métricas do modelo.
"""
    (FINAL_DIR / "README.md").write_text(readme, encoding="utf-8")


def calculate_feature_lift(frame: pd.DataFrame) -> pd.DataFrame:
    base_rate = frame["relevant_flare"].mean()
    records: list[dict[str, float | int | str]] = []

    for column in FEATURE_COLUMNS:
        grouped = frame.groupby(column)["relevant_flare"].agg(["mean", "count"]).reset_index()
        for _, row in grouped.iterrows():
            count = int(row["count"])
            if count < 15:
                continue
            records.append(
                {
                    "feature": column,
                    "value": str(row[column]),
                    "feature_value": f"{column}={row[column]}",
                    "event_rate": float(row["mean"]),
                    "count": count,
                    "lift": float(row["mean"]) / max(base_rate, 1e-12),
                }
            )

    return pd.DataFrame(records).sort_values(["lift", "count"], ascending=[False, False]).head(6)


def copy_assets_to_final() -> None:
    final_data_dir = FINAL_DIR / "data"
    final_figures_dir = FINAL_DIR / "figures"
    if final_data_dir.exists():
        shutil.rmtree(final_data_dir)
    if final_figures_dir.exists():
        shutil.rmtree(final_figures_dir)
    shutil.copytree(DATA_DIR, final_data_dir)
    shutil.copytree(FIGURE_DIR, final_figures_dir)
    shutil.copy2(Path(__file__), FINAL_DIR / "gsia_solar_flare_analysis.py")


def main() -> None:
    prepare_dirs()
    frame = load_solar_flare_data()
    train, test = stratified_split(frame, "relevant_flare", 0.25)

    model = CategoricalNaiveBayes(alpha=1.0)
    model.fit(train, train["relevant_flare"])
    y_score = model.predict_proba_positive(test)
    y_true = test["relevant_flare"].to_numpy()
    metrics = choose_threshold(y_true, y_score)

    risk_table = (
        frame.groupby("zurich_class")["relevant_flare"]
        .agg(event_rate="mean", total="count", events="sum")
        .reset_index()
        .sort_values("event_rate", ascending=False)
    )
    feature_table = calculate_feature_lift(frame)

    summary: dict[str, float | int | str] = {
        "rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "positive_events": int(frame["relevant_flare"].sum()),
        "positive_rate": float(frame["relevant_flare"].mean() * 100),
        "source": "UCI Machine Learning Repository - Solar Flare Dataset",
        "question": "Prever/priorizar risco de flares solares M ou X em 24 horas.",
    }

    create_figures(frame, metrics, risk_table, feature_table)
    copy_assets_to_final()
    write_supporting_files(frame, risk_table, feature_table, metrics, summary)
    build_report(summary, metrics)
    build_presentation(summary, metrics)

    with (FINAL_DIR / "gsia_resumo_execucao.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
        writer.writerow(["threshold", metrics.threshold])
        writer.writerow(["accuracy", metrics.accuracy])
        writer.writerow(["precision", metrics.precision])
        writer.writerow(["recall", metrics.recall])
        writer.writerow(["f2", metrics.f2])

    print(json.dumps({"status": "ok", "final_dir": str(FINAL_DIR), "metrics": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
