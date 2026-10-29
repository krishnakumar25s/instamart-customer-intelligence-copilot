# Instamart Customer Intelligence & Support Copilot
## Project Evaluation Report

## 1. Objective

The objective of this project is to demonstrate an integrated workflow for customer analytics, customer lapse prediction, support-ticket classification, and policy-guided refund assistance using Python, machine learning, SQLite, and information retrieval.

The evaluation covers four components: data preparation, customer intelligence, support-ticket triage, and refund assistance.

## 2. Part A — Data Preparation and Validation

### Dataset Summary

The data-generation script produced the following initial datasets:

| Dataset | Record count |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Raw orders | 13,084 |
| Support tickets | 164 |

### Processing Workflow

The following scripts were used:

- `generate_data.py` — generates the project datasets.
- `audit_data.py` — checks data quality.
- `clean_data.py` — cleans the datasets.
- `validate_data.py` — validates the processed data.

Cleaned datasets are stored under `data/clean`, with the SQLite database at `data/instamart.db`.

**Evaluation outcome:** The data preparation and validation checkpoints were reported as completed during development.

## 3. Part B — Customer Intelligence

### 3.1 K-Means Customer Segmentation

The K-Means evaluation compared cluster counts from 2 through 8 using inertia and silhouette score.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.761 | 0.4650 |
| 3 | 1268.235 | 0.4452 |
| 4 | 743.071 | 0.5255 |
| 5 | 617.366 | 0.4805 |
| 6 | 503.170 | 0.4819 |
| 7 | 408.460 | 0.4163 |
| 8 | 366.503 | 0.4123 |

**Finding:** K = 4 achieved the highest silhouette score (0.5255) among the tested configurations. This supports selecting four clusters based on the silhouette metric for this experiment.

Inertia decreases as the number of clusters increases, which is expected. The selection should therefore consider both cluster compactness and separation rather than inertia alone.

The generated customer-segmentation output is `outputs/customer_segments.csv`. The RFM analysis output is `outputs/rfm_table.csv`.

### 3.2 Customer Lapse Prediction

The project generated a lapse-prediction model and saved it as `outputs/lapse_model.joblib`.

The feature-importance results were:

| Feature | Importance |
|---|---:|
| Frequency | 0.2796 |
| Monetary | 0.2712 |
| Recency | 0.1151 |
| Average delivery minutes | 0.0613 |
| Cancellation rate | 0.0523 |
| Average discount percentage | 0.0484 |
| Favourite category: Electronics | 0.0459 |
| Tenure in days | 0.0447 |
| Age | 0.0377 |

Frequency and monetary value were the two most important features in the saved feature-importance output, followed by recency.

These scores describe the model's relative feature importance. They do not demonstrate that a feature causes a customer to lapse.

### 3.3 Retention Recommendations

The output `outputs/retention_list.csv` contains ten customers ranked by lapse probability.

The highest listed probability was 0.4910 for customer `C0862`. The output also includes an `actually_lapsed` column, allowing the predicted risk to be compared with the observed label.

The retention list can be used to demonstrate how a business might prioritize customers for further review or retention campaigns. The probabilities should not be interpreted as proven business outcomes.

### 3.4 Model Evaluation Limitation

The available outputs establish the clustering scores and feature importance, but they do not provide enough information in this report to state the lapse model's final accuracy, precision, recall, F1-score, or ROC-AUC.

Those metrics should be copied from the actual model-evaluation output before making claims about predictive performance.

## 4. Part C — Support-Ticket Classification

The ticket-triage experiment compared zero-shot and few-shot classification using the supplied deterministic mock LLM.

### Classification Results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets classified | 147 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 11,428 | 22,348 |
| Tokens per classified ticket | 77.74 | 152.03 |

### Interpretation

The few-shot configuration achieved higher measured accuracy on this evaluation dataset but used substantially more tokens.

The zero-shot configuration used fewer tokens per classified ticket, making it more token-efficient in this experiment.

These measurements are specific to the generated evaluation data and the deterministic mock model. They should not be treated as evidence of equivalent performance from a live LLM on real customer tickets.

### PII Redaction and Error Handling

The recorded PII-redaction counts were:

| PII type | Matches detected |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total matches** | **278** |

The experiment also recorded 19 transient errors that were recovered through retry handling.

These are detected-match counts; they do not, by themselves, establish that every sensitive value was detected or that no false positives occurred.

### Token-Budget Experiment

The budget demonstration used a nominal token budget of 6,000. The recorded usage was 6,021 tokens, with 78 tickets classified.

The usage exceeded the nominal limit by 21 tokens. This indicates that the current budget-check behavior should be reviewed if a strict hard limit is required.

Relevant outputs include:

- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

## 5. Part D — Policy Retrieval and Refund Agent

The refund-support component uses SQLite order lookups, policy retrieval, refund eligibility rules, and guarded tool execution.

### Policy Retrieval

The policy retrieval implementation uses TF-IDF vectorization and cosine similarity. The configured retrieval process uses 40-word chunks, a 10-word overlap, and top-three retrieval results.

Retrieved passages provide supporting policy context for refund decisions.

### Refund Rules

The demonstrated business rules are:

1. Only completed orders can receive a late-delivery refund.
2. Orders delivered in more than 45 minutes qualify under the late-delivery rule.
3. Orders delivered in 45 minutes or less do not qualify.
4. Refunds of ₹1,000 or less can follow the simulated automatic-refund path.
5. Refunds above ₹1,000 require human approval.
6. Missing or unknown order information must be handled safely.

### Scenario Evaluation

| Scenario | Observed result |
|---|---|
| Completed order delivered in 20 minutes | Refund denied |
| Unknown order ID | Order not found |
| Missing order ID | Requests additional information |
| Completed order delivered in 74 minutes, amount ₹12 | Simulated refund issued |
| Completed order delivered in 62 minutes, amount ₹2,598 | Escalated to a human lead |
| Cancelled order | Refund denied |
| Repeated tool call | Loop detected and stopped |

The successful automatic-refund test returned `status: issued` for order `ORD-107374`. The high-value test returned `status: escalated` for order `ORD-107652`.

The cancelled-order test rejected `ORD-102006` because its status was `cancelled`.

### Safety and Limitations

The tested scenarios demonstrate order validation, eligibility checking, human escalation, and loop detection.

Refund actions are simulated. The project does not process actual payments or modify a real customer's financial account. The supplied `mock_llm.py` is deterministic and is not a live LLM service.

The scenario tests demonstrate selected paths; they are not a substitute for comprehensive automated testing or production security review.

## 6. Overall Evaluation

| Component | Evaluation evidence |
|---|---|
| Data preparation | Generation, cleaning, auditing, and validation checkpoints completed |
| Customer segmentation | Best tested silhouette score: 0.5255 at K = 4 |
| Lapse prediction | Model and feature-importance artifacts generated; final predictive metrics require verification |
| Ticket classification | 85.71% zero-shot accuracy and 100% few-shot accuracy on the recorded experiment |
| PII handling | 278 detected matches across email, phone, and order ID patterns |
| Error handling | 19 recorded transient errors recovered |
| Policy retrieval | Top-three TF-IDF/cosine-similarity retrieval demonstrated |
| Refund agent | Automatic refund, human escalation, cancellation rejection, unknown-order handling, and loop detection demonstrated |

## 7. Conclusion

The project demonstrates how customer analytics and support automation can be combined into a multi-component Python application.

The segmentation experiment identifies K = 4 as the strongest tested configuration according to silhouette score. The ticket-classification experiment illustrates a trade-off between token usage and measured accuracy. The refund agent demonstrates policy-guided decisions and safeguards across selected scenarios.

Further work should include verification of the lapse model's predictive metrics, broader automated testing, stronger PII evaluation, strict token-budget enforcement, and integration with real services only where appropriate authorization and security controls are available.

The current implementation is a demonstration built on generated data and a deterministic mock LLM, not a production customer-support system.
