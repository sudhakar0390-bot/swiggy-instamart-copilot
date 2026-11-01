# Swiggy Instamart Customer Intelligence & Support Copilot

## Overview

This project develops a Customer Intelligence & Support Copilot for Swiggy Instamart using Python, pandas, SQLite, scikit-learn, and the supplied offline mock LLM.

The project contains four connected parts:

- **Part A:** Data cleaning, analytical database, and SQL analysis
- **Part B:** Customer segmentation and lapse prediction
- **Part C:** Support ticket classification, privacy protection, and routing
- **Part D:** Policy RAG assistant and refund agent

## Reproduce

Run the following commands from the project root after placing the supplied `generate_data.py` and `mock_llm.py` files there.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py
python part_a\clean_data.py
python part_a\run_queries.py
python part_a\analyze_data.py
python part_a\validate_data.py
```

Commands for Parts B, C, and D will be added after implementation.

## Part A — Data Foundation

### Data Cleaning

The raw datasets contained 1,200 customers, 40 products, and 13,084 order rows.

Cleaning included:

- Removing duplicate order records and excluding pending orders.
- Standardizing customer city and gender values.
- Converting invalid ages to missing values.
- Repairing invalid order amounts using product prices, quantities, and discounts.
- Converting negative delivery times into missing values.
- Saving cleaned datasets and loading them into SQLite.

### Validated Results

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Completed-order revenue | ₹29,98,668.80 |
| SQL–pandas revenue difference | ₹0.00 |

The database contains three tables: `customers`, `products`, and `orders`.

### SQL Business Findings

Seven SQL queries were executed.

- **Highest-revenue city:** Hyderabad — ₹5,28,589.90.
- **Lowest-revenue city:** Chennai — ₹4,63,629.15.
- **Highest-revenue category:** Electronics — ₹12,19,248.10.
- **Highest-spending customer:** C0693 — ₹14,766.55.
- **Highest-revenue month:** April 2025 — ₹5,33,690.05.

SQL and pandas produced identical completed-order revenue totals.

### Outlier Analysis

The IQR method was applied to completed-order amounts.

| Metric | Value |
|---|---:|
| Q1 | ₹64.80 |
| Q3 | ₹297.00 |
| IQR | ₹232.20 |
| Lower boundary | −₹283.50 |
| Upper boundary | ₹645.30 |
| Outliers detected | 897 |
| Outlier percentage | 8.07% |

Outliers were retained because unusually high order amounts do not necessarily represent incorrect transactions.

### Correlation Analysis

- Quantity vs. order amount: 0.087.
- Discount percentage vs. order amount: −0.095.
- Delivery minutes vs. order amount: −0.003.

These results indicate weak linear relationships. Among completed orders, 774 delivery-time values were missing.

### Validation

Ten automated checks passed, covering record counts, order statuses, unique order IDs, revenue, missing order amounts, and database table existence.

Final validation output:

`ALL CHECKPOINTS PASSED`

## Part B — Customer Segments and Lapse Prediction

## Part B — Customer Segmentation and Lapse Prediction

### B1. RFM Customer Segmentation

RFM (Recency, Frequency, Monetary) features were calculated from completed customer orders.

- Total customers: 1,200
- Average recency: 37.62 days
- Average frequency: 9.26 orders
- Average monetary value: INR 2,498.89

K-Means clustering was performed using standardized RFM features. Values of K from 2 to 8 were evaluated using silhouette scores.

**Selected K: 4**

**Silhouette score: 0.5394**

| Segment | Customers | Avg. Recency | Avg. Frequency | Avg. Monetary (INR) |
|---|---:|---:|---:|---:|
| Premium | 145 | 26.23 | 6.68 | 6,988.18 |
| Loyal | 419 | 9.75 | 19.40 | 3,410.02 |
| Regular | 469 | 33.46 | 3.70 | 1,006.40 |
| At-Risk | 167 | 129.09 | 1.65 | 506.47 |

Premium customers have the highest average spending, Loyal customers order most frequently, Regular customers have lower purchasing activity, and At-Risk customers have long periods since their last completed order.

PCA was applied to the standardized RFM features:

- PC1 explained variance: 64.43%
- PC2 explained variance: 20.94%
- Combined explained variance: 85.37%

Outputs: `outputs/rfm_customers.csv`, `outputs/customer_segments.csv`, and `outputs/customer_segments_pca.csv`.

### B2. Customer Lapse Prediction

A time-based prediction dataset was created using a cutoff date of September 1, 2025. Only completed orders before the cutoff were used to construct features. September activity was used to determine whether customers lapsed.

- Eligible customers: 1,156
- Active customers: 713
- Lapsed customers: 443
- Training customers: 924
- Testing customers: 232

Features included age, city, gender, historical order frequency, monetary value, average discount, average quantity, and recency.

Preprocessing included missing-value imputation, numeric standardization, and categorical one-hot encoding within scikit-learn pipelines.

Two classifiers were tuned using five-fold cross-validation:

| Model | Cross-validation F1 | Test F1 |
|---|---:|---:|
| Logistic Regression | 0.6508 | 0.6531 |
| Random Forest | 0.6909 | 0.6484 |

Random Forest was selected based on cross-validation performance.

Best Random Forest parameters:

- `max_depth=5`
- `min_samples_leaf=10`
- `n_estimators=200`

The selected model achieved a test F1 score of **0.6484**, exceeding the required minimum of 0.55.

Confusion matrix:

```text
[[109, 34],
 [ 30, 59]]
```

### B3. Class Imbalance Experiment

The original training data contained 570 active and 354 lapsed customers.

Random oversampling was applied only to the training set, resulting in 570 examples per class. The test set was not modified.

| Experiment | Test F1 |
|---|---:|
| Original Random Forest | 0.6484 |
| Oversampled Random Forest | 0.6754 |

Oversampling improved the observed test F1 by 0.0270 and increased lapse recall, but reduced overall accuracy. The original cross-validation-selected model remains the saved final model; the oversampling comparison is exploratory.

### B4. Premium Customer Retention List

The test predictions were joined with the customer segmentation results using customer IDs.

- Premium customers in the test set: 41
- Customers shortlisted: 10
- Highest predicted lapse probability: 53.29%
- Highest-ranked customer: C0060

The top 10 Premium customers were ranked by predicted lapse probability.

Output: `outputs/retention_list.csv`.

### B5. Limitations

- Customer segment names are mapped to the cluster numbers from the current reproducible K-Means run.
- Full-history RFM segments include September activity, so the Premium retention list is illustrative rather than a leakage-free historical evaluation.
- Oversampling was evaluated on the existing test set as an additional experiment and was not used to select the saved model.
- Model performance may change with different datasets or future customer behavior.

## Part C — Support Ticket Triage with Guardrails

## Part C — Support Ticket Triage and Privacy Protection

### C1. Ticket Data Inspection

The generated dataset contains 164 customer support tickets with five columns: `ticket_id`, `customer_id`, `order_id`, `text`, and `true_category`.

- Total tickets: 164
- Empty ticket messages: 4
- Empty order IDs: 8
- Duplicate rows: 0

The `true_category` column was reserved for evaluation and was not provided to the classifier as input.

### C2. PII Redaction

Regular expressions were used to remove sensitive information from ticket messages before classification.

| PII type | Redacted occurrences |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

Sensitive information was replaced with `[EMAIL]`, `[PHONE]`, and `[ORDER_ID]`.

The classification system uses redacted message text rather than the original ticket text.

Output: `outputs/redacted_tickets.csv`.

### C3. Ticket Classification

The supplied offline `mock_llm.py` was used through its public `chat()` interface, without external APIs or paid services.

The classification system includes JSON parsing and validation, temporary-error retries with exponential backoff, keyword fallback for non-JSON responses, and manual-review flags for invalid structured outputs.

| Classification status | Tickets |
|---|---:|
| Valid JSON classification | 135 |
| Keyword fallback | 12 |
| Invalid output flagged | 9 |
| Empty text flagged | 4 |
| Excessively long text flagged | 4 |
| **Total** | **164** |

Successfully classified tickets: 147.

Classification accuracy among successfully classified tickets: **85.71%**.

Reported token usage: 7,528 tokens.

Output: `outputs/triage_results.csv`.

### C4. Zero-Shot vs Few-Shot Comparison

Two prompting strategies were evaluated using the same redacted tickets.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.71% | 100% |

Few-shot prompting improved accuracy by 14.29 percentage points in the deterministic offline mock evaluation.

The 100% accuracy result is specific to this generated dataset and mock model and should not be interpreted as guaranteed real-world performance.

### C5. Token-Budget Guardrails

A token-budget demonstration estimated prompt usage and reserved 100 tokens for responses.

| Token budget | Estimated prompt tokens | Actual tokens | Result |
|---|---:|---:|---|
| 20 | 45 | 0 | Blocked |
| 100 | 45 | 0 | Blocked |
| 500 | 45 | 52 | Within budget |

Requests that exceeded the estimated budget were blocked before invoking the mock LLM.

Output: `outputs/triage_budget_demo.csv`.

### C6. Customer-Segment-Based Routing

Customer segment information from Part B was joined with ticket classification results using `customer_id`.

Customer segment distribution among tickets:

- Loyal: 66
- Regular: 62
- At-Risk: 20
- Premium: 16

Routing priority distribution:

| Priority | Tickets |
|---|---:|
| Medium | 112 |
| Critical | 23 |
| Review | 17 |
| High | 12 |

Support queue distribution:

| Support queue | Tickets |
|---|---:|
| Billing Support | 53 |
| Delivery Support | 33 |
| Technical Support | 26 |
| Fraud Investigation | 23 |
| Manual Review | 17 |
| Priority Support | 12 |

Fraud classifications were routed to critical-priority investigation. Premium customers with other successfully classified issues received high-priority support. Unclassified tickets were routed for manual review.

Output: `outputs/routed_tickets.csv`.

### C7. Limitations

- The supplied offline mock LLM is deterministic and does not represent the full behavior of a production LLM.
- Keyword fallback can produce incorrect classifications.
- Invalid structured outputs require manual review.
- The token-budget demonstration uses estimated token counts and a fixed response reserve.
- Routing rules are illustrative business rules and should be validated against operational requirements before deployment.
- PII redaction covers the specified email, phone, and order ID patterns; it is not a comprehensive personal-information detection system.

## Part D — Policy RAG Assistant and Refund Agent

### 1. Policy Retrieval-Augmented Generation (RAG)

Implemented an offline policy question-answering system using TF-IDF vectorization and cosine similarity.

- Policy documents: 5
- Total policy chunks: 15
- Chunk size: 40 words
- Chunk overlap: 10 words
- Retrieval: Top 3 most relevant chunks
- LLM: Provided offline `mock_llm.py`

**Policy RAG evaluation results:**

| Metric | Result |
|---|---|
| Evaluation questions | 14 |
| Supported questions | 12 |
| Retrieval success | 12/12 (100%) |
| Factual answer success | 12/12 (100%) |
| Unsupported-question fallback | 2/2 (100%) |

Evaluation results are saved in `outputs/policy_eval_results.csv`.

### 2. Refund Agent

Implemented a ReAct-style refund support agent using the provided offline mock LLM.

The agent uses five tools:

1. `get_order` — Retrieves order information from the SQLite database.
2. `search_policy` — Retrieves relevant refund-policy passages.
3. `calculate_refund` — Calculates refund eligibility and amount.
4. `issue_refund` — Processes eligible automatic refunds.
5. `escalate_to_human` — Sends high-value refunds for human approval.

**Refund policy rules:**

- Only completed orders are eligible for a refund.
- Late-delivery refunds require delivery time greater than 45 minutes.
- Eligible orders receive a full refund.
- Refunds up to INR 1,000 can be processed automatically.
- Refunds above INR 1,000 require human approval.

### 3. Refund Agent Test Results

| Scenario | Actual result |
|---|---|
| Eligible late-delivery order | Automatic refund of INR 75.60 |
| Eligible high-value order | INR 1,098.00 sent for human approval |
| Cancelled order | DENIED |
| Delivery within 45 minutes | DENIED |
| Unknown order ID | NOT_FOUND |
| Missing order ID | NEEDS_INFO |
| Repeated refund attempt | Duplicate prevented |

Refund records are saved in `outputs/refund_ledger.jsonl`.

Human escalations are saved in `outputs/refund_escalations.jsonl`.

### 4. Agent Safety Limits

Implemented and tested the following safeguards:

| Safety mechanism | Configuration | Test result |
|---|---|---|
| Maximum iterations | 6 | PASS |
| Token budget | 6,000 tokens | PASS using a 1-token test budget |
| Execution timeout | 10 seconds | PASS using a 0-second test timeout |

The agent returns fallback responses when its configured limits are reached.

### 5. Execution Logging

Implemented JSONL execution logging in `outputs/agent_runs.jsonl`.

Each normally completed run records the customer request, final response, total token usage, tool names, tool arguments, and tool results.

A verified on-time-delivery run used 680 tokens and executed three tools.

### 6. Checkpoint and Resume

Implemented checkpoint saving and restoration using `part_d/checkpoint.py`.

The checkpoint stores conversation messages, completed iteration count, token usage, and tool execution trace.

A simulated interruption test confirmed that the agent resumed from iteration 2 without repeating the completed `get_order` call. The checkpoint was cleared after successful completion.

This verifies recovery between completed iterations, not crash-safe recovery during an in-progress refund operation.

### 7. ReAct vs Plan-and-Execute

The project
## Limitations and Next Steps

The project uses synthetic data. Results may not reflect real customer behavior. The current checks validate key dataset and revenue properties but do not yet cover every cleaning and data-quality requirement.

Next steps include completing the remaining data-quality checks, building customer segments, predicting customer lapse, implementing support ticket routing, and developing the policy assistant and refund agent.
