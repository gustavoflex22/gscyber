# GSIA - IA e Machine Learning com dados espaciais

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

# GitHub Actions

Este repositório inclui o workflow `Security Scan`, que executa o job `Trivy Vulnerability Scan` em pushes, pull requests e execuções manuais.

Para simular uma falha de segurança, execute o workflow manualmente pela aba Actions com `simulate_failure=true`. Essa opção cria um fixture vulnerável temporário para que o Trivy detecte vulnerabilidades `HIGH/CRITICAL` e bloqueie o pipeline.
