# AGENTS.md - Adaptive Blood Infrastructure System (ABIS)

## System Overview
The Adaptive Blood Infrastructure System (ABIS) is an intelligent, resilient blood-management platform engineered for African healthcare realities. It connects blood donor drives, testing laboratories, regional cold rooms, and hospital transfusion centers to eliminate inventory blind spots and guarantee cold-chain and supply traceability.

---

## Operational Rules & Boundaries

### 1. Security & Credentials
- **Strict Prohibition:** Under no circumstances should `.env` files, production credentials, database passwords, or any file containing plaintext secrets be committed to version control or printed directly in logs.
- Always use environment variables (e.g., `DATABASE_URL`) with runtime configuration loading.
- Ensure all sensitive variables are listed in `.gitignore`.

### 2. Code Quality & Typing
- **Type Checking:** Enforce clean, modular Python code with strict type annotations compatible with Pylance and standard typing systems (`typing.Optional`, `typing.List`, `typing.Dict`, `pydantic.BaseModel`).
- **Separation of Concerns:** Keep routing logic (`routes.py`), ORM database definitions (`models.py`), database session lifecycle (`database.py`), and machine learning algorithms (`ml_engine.py`) strictly isolated.
- **Defensive Error Handling:** Handle missing connections, empty datasets, and model inference errors gracefully with informative HTTP status codes and structured logging.

### 3. Naming Conventions
- **Variables & Functions:** Strict `snake_case` (e.g., `units_requested`, `generate_donor_ledger`, `calculate_retention_score`).
- **Classes & Pydantic/SQLAlchemy Models:** Strict `PascalCase` (e.g., `DonorProfile`, `TransfusionDemand`, `RetentionPredictionRequest`).
- **Domain Specificity:** Use descriptive, medically accurate domain terminology (e.g., `TransfusionRequest`, `syphilis_s_co_ratio`, `recency_days`, `frequency_total`, `cold_chain_breach`). Avoid generic names like `data`, `item`, or `temp`.

### 4. Machine Learning & Mathematical Reasoning
- **Chain-of-Thought Reasoning:** Explicitly state and document the mathematical rationale, statistical distributions, and clinical priors behind ML implementations:
  - *Donor Retention:* Modeled via Random Forest on RFM (Recency, Frequency, Tenure) behavioral vectors.
  - *Serology Screening Ratios:* Based on clinical research where Signal-to-Cutoff ($s/co$) $\ge 10$ yields a 98.4% positive predictive value.
  - *Transfusion Demand:* Modeled via mean-reverting stochastic random walk processes: $D_t = D_{t-1} + \theta(\mu - D_{t-1}) + \sigma \epsilon_t$, with ARIMA/SARIMA forecasting for regional rebalancing.
- **Fail-Safe Fallbacks:** Ensure algorithmic predictions have deterministic heuristic fallbacks if model inference fails or sample sizes are insufficient.
