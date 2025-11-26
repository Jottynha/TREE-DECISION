# Makefile para Trabalho Prático de IA
# =====================================
# Automatiza execução de todas as partes do projeto

.PHONY: help install clean part1 part2 part2-preprocess part2-dt part2-knn part2-svm part2-all part3 results

# Cores para output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)║     MAKEFILE - TRABALHO PRÁTICO DE IA (2025/2)              ║$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Comandos Disponíveis:$(NC)"
	@echo ""
	@echo "  $(YELLOW)make install$(NC)          - Instala todas as dependências (roda no run.sh)"
	@echo "  $(YELLOW)make part1$(NC)            - Executa Parte 1 (Árvore Manual Filosófica)"
	@echo "  $(YELLOW)make part2$(NC)            - Executa Parte 2 completa (ML: pré-proc + treinos)"
	@echo "  $(YELLOW)make part2-preprocess$(NC) - Apenas pré-processamento dos dados"
	@echo "  $(YELLOW)make part2-dt$(NC)         - Treina apenas Decision Tree"
	@echo "  $(YELLOW)make part2-knn$(NC)        - Treina apenas KNN"
	@echo "  $(YELLOW)make part2-svm$(NC)        - Treina apenas SVM"
	@echo "  $(YELLOW)make part3$(NC)            - Executa Parte 3 (Algoritmo Genético)"
	@echo "  $(YELLOW)make results$(NC)          - Exibe relatório de resultados"
	@echo "  $(YELLOW)make clean$(NC)            - Remove arquivos processados e modelos"
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"

# ============================================================
# INSTALAÇÃO E CONFIGURAÇÃO
# ============================================================

install:
	@echo "$(GREEN)Instalando dependências...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependências instaladas com sucesso!$(NC)"

# ============================================================
# PARTE 1: ÁRVORE DE DECISÃO MANUAL (FILOSÓFICA)
# ============================================================

part1:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)║  PARTE 1: ÁRVORE DE DECISÃO MANUAL (FILOSÓFICA)             ║$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@cd src/part1_tree_manual && python3 tree_manual.py
	@echo "$(GREEN)✓ Parte 1 concluída!$(NC)"

# ============================================================
# PARTE 2: MACHINE LEARNING (SUPERVISIONADO)
# ============================================================


part2-dt:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)║  TREINAMENTO: DECISION TREE                                 ║$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@python3 src/part2_ml/train_tree.py
	@echo "$(GREEN)✓ Decision Tree treinada!$(NC)"

part2-knn:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)║  TREINAMENTO: KNN (K-NEAREST NEIGHBORS)                     ║$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@python3 src/part2_ml/train_knn.py
	@echo "$(GREEN)✓ KNN treinado!$(NC)"

part2-svm:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)║  TREINAMENTO: SVM (SUPPORT VECTOR MACHINE)                  ║$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@python3 src/part2_ml/train_svm.py
	@echo "$(GREEN)✓ SVM treinado!$(NC)"

part2-all: part2-dt part2-knn part2-svm
	@echo "$(GREEN)✓ Todos os modelos treinados!$(NC)"

part2: part2-preprocess part2-all
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ PARTE 2 CONCLUÍDA COM SUCESSO!$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Resultados salvos em:$(NC)"
	@echo "  - data/processed/benchmark_results.csv"
	@echo "  - data/processed/comparison_report.txt"
	@echo "  - data/processed/confusion_matrix_*.png"
	@echo ""

# ============================================================
# PARTE 3: ALGORITMO GENÉTICO
# ============================================================

part3:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)║  PARTE 3: ALGORITMO GENÉTICO                                ║$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@cd src/part3_ga && python3 ga.py
	@echo "$(GREEN)✓ Parte 3 concluída!$(NC)"

# ============================================================
# VISUALIZAÇÃO DE RESULTADOS
# ============================================================

results:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)║  RELATÓRIO DE RESULTADOS - PARTE 2 (ML)                     ║$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@if [ -f data/processed/comparison_report.txt ]; then \
		cat data/processed/comparison_report.txt; \
	else \
		echo "$(RED)✗ Relatório não encontrado. Execute 'make part2' primeiro.$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Arquivos gerados:$(NC)"
	@ls -lh data/processed/*.csv data/processed/*.png 2>/dev/null || echo "$(RED)Nenhum arquivo encontrado$(NC)"

# ============================================================
# LIMPEZA
# ============================================================

clean:
	@echo "$(RED)Removendo arquivos processados...$(NC)"
	rm -f data/processed/*.csv
	rm -f data/processed/*.png
	rm -f data/processed/*.txt
	rm -f svm.model
	rm -rf src/part2_ml/__pycache__
	rm -rf src/part2_ml/data/processed/*
	@echo "$(GREEN)✓ Arquivos removidos!$(NC)"

# ============================================================
# EXECUÇÃO COMPLETA (TODAS AS PARTES)
# ============================================================

all: part1 part2 part3
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓✓✓ TRABALHO COMPLETO EXECUTADO COM SUCESSO! ✓✓✓$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"