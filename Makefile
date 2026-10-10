# =============================================================================
# MoE Demand Forecasting - Makefile
# Uso: make <target>
# Todo corre dentro de Docker: no hace falta instalar Python en tu máquina.
# =============================================================================

.PHONY: help build up down pipeline benchmark nb clean

COMPOSE := docker compose
# Ejecuta un notebook con papermill y guarda la copia ejecutada en outputs/notebooks/
RUN_NB = $(COMPOSE) run --rm runner papermill --cwd /app/notebooks --log-output \
	/app/notebooks/$(1).ipynb /app/outputs/notebooks/$(1).ipynb

NOTEBOOKS := 01_data_acquisition 02_preprocessing 03_clustering 04_cluster_analysis_classifiers \
	05_expert_selection 06_gating_design 07_hpo 08_production_assembly

## help: Muestra esta ayuda
help:
	@echo "MoE Demand Forecasting - Comandos disponibles:"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  make /'

## build: Construye la imagen Docker con las dependencias fijadas
build:
	@$(COMPOSE) build

## up: Levanta JupyterLab en http://localhost:8888
up:
	@$(COMPOSE) up -d --build --wait jupyter
	@echo "  [OK] JupyterLab en http://localhost:8888"

## down: Detiene JupyterLab
down:
	@$(COMPOSE) down

## pipeline: Ejecuta los notebooks 01 a 08 en orden (datos → modelo de producción)
pipeline:
	@mkdir -p outputs/notebooks
	@for nb in $(NOTEBOOKS); do \
		echo "==> $$nb"; \
		$(call RUN_NB,$$nb) || exit 1; \
	done
	@echo "  [OK] Modelo de producción en models/production/"

## benchmark: Ejecuta el notebook 09 (MoE vs. modelos de referencia; tarda varias horas en CPU)
benchmark:
	@mkdir -p outputs/notebooks
	@$(call RUN_NB,09_benchmark_comparison)

## nb: Ejecuta un solo notebook (make nb NB=03_clustering)
nb:
	@test -n "$(NB)" || (echo "Uso: make nb NB=<nombre del notebook sin .ipynb>" && exit 1)
	@mkdir -p outputs/notebooks
	@$(call RUN_NB,$(NB))

## clean: Borra datos, modelos, resultados y notebooks ejecutados (se regeneran con make pipeline)
clean:
	@find data models results -type f ! -name '.gitignore' ! -name '.gitkeep' -delete
	@rm -rf outputs
	@echo "  [OK] Limpieza completada"
