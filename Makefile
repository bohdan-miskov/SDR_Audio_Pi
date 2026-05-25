.PHONY: help lint check-deps check req req-dev sync deps

help:
	@echo "Available commands:"
	@echo "  make check      - Run all checks (Ruff, MyPy, Deptry)"
	@echo "  make fix        - Run small fixes with Ruff"
	@echo "  make lint       - Run linter and type checker only"
	@echo "  make deps       - Update and sync all dependencies (req + req-dev + sync)"
	@echo "  make req        - Compile production dependencies only"
	@echo "  make req-dev    - Compile development dependencies"
	@echo "  make sync       - Install dependencies to local .venv"

# --- ПЕРЕВІРКА КОДУ ---

# Запускає Ruff (лінтер) та MyPy (типи)
fix:
	ruff check . --fix

lint:
	ruff check .
	mypy . 

# Запускає Deptry для перевірки залежностей
check-deps:
	deptry .

# Головна команда перевірки 
check: lint check-deps


# --- РОБОТА ІЗ ЗАЛЕЖНОСТЯМИ (pip-tools) ---

# 1. Компілюємо основні залежності
req:
	pip-compile requirements.in

# 2. Компілюємо dev залежності
req-dev:
	pip-compile requirements-dev.in

# 3. Синхронізуємо локальне середовище (.venv)
sync:
	pip-sync requirements.txt requirements-dev.txt

# 4. Все разом: повне оновлення середовища однією командою
deps: req req-dev sync