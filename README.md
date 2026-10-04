# FinSight

## Hybrid AI Customer Support Intelligence System
## 🚀 Live Demo

[Try FinSight Live](https://finsight-test.streamlit.app)

FinSight is a hybrid customer-support intelligence system designed for banking-related customer queries.

The system combines a traditional Machine Learning classifier with a Large Language Model (LLM). Straightforward queries are handled using a Linear SVM model, while ambiguous queries are routed to an LLM for deeper semantic analysis.

The goal is to achieve reliable intent classification while reducing unnecessary LLM usage, latency, and API calls.

---

## Problem Statement

Banking customer-support systems receive a large number of queries covering different issues such as card problems, transfers, identity verification, payments, refunds, cash withdrawals, and account-related problems.

A traditional intent classifier can efficiently handle straightforward queries but may struggle when multiple intents are semantically similar.

On the other hand, using an LLM for every query can introduce additional latency and unnecessary API usage.

FinSight addresses this problem using a hybrid architecture:

**Customer Query → Domain Check → ML Classification → Confidence Check → ML Response / LLM Fallback**

---

## Key Features

- Banking customer-support intent classification
- TF-IDF based text representation
- Linear SVM intent classifier
- Confidence-based ML/LLM routing
- Groq LLM integration
- Structured JSON responses
- Pydantic-based response validation
- BANKING77 intent restriction
- Banking-domain guard
- Basic spelling-error detection
- Streamlit user interface
- ML vs Hybrid performance experiment

---

## System Architecture

```text
                    Customer Query
                          |
                          v
                 Domain Guard Check
                    /           \
             Non-Banking       Banking
                  |                |
                  v                v
             Out of Scope     Text Cleaning
                                   |
                                   v
                              TF-IDF Vectorizer
                                   |
                                   v
                              Linear SVM
                                   |
                                   v
                         Calculate SVM Margin
                                   |
                         +---------+---------+
                         |                   |
                    High Margin          Low Margin
                         |                   |
                         v                   v
                    ML Prediction       Groq LLM
                                             |
                                             v
                                      Structured JSON
                                             |
                                             v
                                      Pydantic Validation
                                             |
                                             v
                                      Support Response
```

---

## Dataset

FinSight uses the BANKING77 dataset.

The dataset contains:

- 13,083 total samples
- 10,003 training samples
- 3,080 test samples
- 77 banking-related customer-support intents

Each sample contains a customer query and its corresponding intent category.

Examples of intents include:

- `verify_my_identity`
- `card_not_working`
- `cash_withdrawal_not_recognised`
- `transfer_not_received_by_recipient`
- `automatic_top_up`
- `refund_not_showing_up`
- `balance_not_updated_after_bank_transfer`

---

## NLP Preprocessing

The text preprocessing pipeline performs the following operations:

1. Convert text to lowercase
2. Remove URLs
3. Remove email addresses
4. Normalize special characters
5. Normalize whitespace

Example:

```text
"I can't verify my identity!"
```

is transformed into a normalized form before being passed to the TF-IDF vectorizer.

---

## Feature Engineering

TF-IDF is used to convert customer queries into numerical feature vectors.

The vectorizer uses:

- Unigrams
- Bigrams
- `min_df=2`
- `max_df=0.95`

The TF-IDF vectorizer is fitted only on the training data and then used to transform the test data.

This prevents information from the test set from influencing the training vocabulary.

---

## Machine Learning Models

Four classification models were evaluated:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.78% | 86.72% | 85.78% | 85.68% |
| Linear SVM | 88.93% | 89.30% | 88.93% | 88.93% |
| Multinomial Naive Bayes | 79.90% | 83.29% | 79.90% | 78.57% |
| Random Forest | 85.81% | 86.53% | 85.81% | 85.81% |

Under the tested configurations, Linear SVM achieved the highest measured performance on the BANKING77 test set.

The trained SVM model and TF-IDF vectorizer are saved using Joblib.

---

## Why Linear SVM?

Linear SVM works well for high-dimensional sparse text features such as TF-IDF.

The model produces decision scores for all 77 intents.

Instead of treating the SVM output as a probability, FinSight uses the difference between the highest and second-highest decision scores as a confidence margin.

```text
Margin = Highest Decision Score - Second Highest Decision Score
```

A larger margin indicates stronger separation between the top two predicted classes.

A smaller margin indicates that the model is less certain between competing intents.

---

## Hybrid ML + LLM Routing

FinSight does not send every query to the LLM.

The system first evaluates the query using the Linear SVM model.

The current routing threshold is:

```text
0.5
```

The routing logic is:

```text
High SVM Margin
      |
      v
ML handles query

Low SVM Margin
      |
      v
LLM handles query
```

The threshold was selected experimentally using the confidence-margin analysis on the test data.

It is a project-specific threshold and is not intended to be a universal confidence threshold for SVM models.

---

## Confidence Threshold Experiment

Different confidence thresholds were evaluated to understand the trade-off between ML coverage and accepted prediction accuracy.

| Threshold | ML Coverage | Accuracy on Accepted Predictions |
|---:|---:|---:|
| 0.1 | 95.16% | 91.95% |
| 0.2 | 91.14% | 93.69% |
| 0.3 | 87.08% | 95.38% |
| 0.4 | 83.64% | 96.20% |
| 0.5 | 79.35% | 97.14% |
| 0.6 | 75.62% | 97.98% |
| 0.7 | 71.82% | 98.42% |
| 0.8 | 67.44% | 98.94% |
| 1.0 | 58.44% | 99.44% |
| 1.2 | 47.99% | 99.53% |

The experiment demonstrates the trade-off between allowing the ML model to handle more queries and requiring stronger separation between competing intents.

---

## LLM Integration

FinSight uses the Groq API with:

```text
Model: openai/gpt-oss-20b
```

The LLM is used only when the ML classifier produces an ambiguous prediction.

The LLM receives the customer query along with instructions defining the expected output format.

The LLM is required to return structured JSON containing:

- Intent
- Sentiment
- Urgency
- Summary
- Entities
- Root cause
- Recommended action
- Department
- Escalation requirement
- Customer response

---

## Intent Restriction

The LLM is restricted to the same 77 intent labels used by BANKING77.

This prevents the LLM from inventing new intent names that are outside the project's classification system.

The returned intent is also checked against the trained SVM model's available intent classes.

---

## Pydantic Validation

The LLM response is parsed from JSON and validated using Pydantic.

This ensures that the response follows the expected structure before it is used by the application.

The validation schema contains:

```text
intent
sentiment
urgency
summary
entities
root_cause
recommended_action
department
escalation_required
customer_response
```

---

## Hybrid Experiment

A separate experiment was conducted using a balanced sample containing:

```text
77 intents × 2 samples per intent = 154 samples
```

### ML-Only

- Samples: 154
- Accuracy: 88.31%
- Processing time: 0.2311 seconds

### Routing Results

Using the confidence threshold of `0.5`:

- Total samples: 154
- ML handled: 118
- LLM fallback candidates: 36
- ML coverage: 76.62%
- LLM fallback rate: 23.38%

### LLM Fallback

Out of the 36 fallback candidates:

- 34 LLM calls completed successfully
- 2 calls were interrupted because of a Groq HTTP 429 rate-limit response
- LLM fallback accuracy on the 34 completed calls: 67.65%
- Average LLM latency: 7.21 seconds
- Minimum latency: 0.63 seconds
- Maximum latency: 13.24 seconds

### Hybrid Result

The combined hybrid experiment achieved:

- ML-only accuracy: 88.31%
- Hybrid accuracy: 88.96%
- Improvement: 0.65 percentage points
- ML coverage: 76.62%

The experiment demonstrates the behavior of the hybrid architecture on the selected balanced sample.

The LLM was not more accurate than the SVM on the ambiguous fallback cases. Its role is to provide deeper semantic analysis and structured support information for cases where the ML model has lower confidence.

---

## Domain Guard

Before classification, FinSight checks whether the query appears to be related to the banking domain.

The domain guard uses a lightweight keyword-based approach containing terms related to:

- Banking
- Cards
- Payments
- Transfers
- Transactions
- Cash withdrawals
- Accounts
- Identity verification
- Currency
- ATMs
- Wallets

Queries that do not appear to be banking-related are returned as:

```text
OUT_OF_SCOPE
```

This prevents unrelated questions from being incorrectly classified into one of the BANKING77 intents.

---

## Spelling Detection

FinSight also performs basic spelling-error detection before analysis.

Common banking-specific terms are excluded from generic spell checking.

For example:

```text
I cant verify my identty
```

produces a warning asking the user to check the query instead of sending the query for analysis.

This is intentionally a lightweight input-quality check rather than a full grammar-correction system.

---

## Streamlit Application

The user interface is built using Streamlit.

The application provides:

- Customer query input
- Spelling validation
- Banking-domain validation
- ML intent classification
- LLM fallback
- Customer-facing support response

### Example ML Flow

```text
Customer Query
      ↓
Domain Check
      ↓
SVM Classification
      ↓
High Confidence
      ↓
Issue Identified
```

### Example LLM Flow

```text
Customer Query
      ↓
Domain Check
      ↓
SVM Classification
      ↓
Low Confidence
      ↓
Groq LLM
      ↓
Structured Analysis
      ↓
Customer Response
```

---

## Project Structure

```text
FinSight/
│
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│
├── prompts/
│   └── support_analysis.txt
│
└── src/
    ├── llm/
    │   ├── __init__.py
    │   └── llm_analyzer.py
    │
    ├── nlp/
    │   ├── __init__.py
    │   └── preprocessing.py
    │
    ├── pipeline/
    │   ├── __init__.py
    │   ├── hybrid_pipeline.py
    │   └── domain_guard.py
    │
    ├── schemas/
    │   ├── __init__.py
    │   └── support_schema.py
    │
    └── utils/
        ├── __init__.py
        └── spelling.py
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aaryanbhatia50-cyber/FinSight.git
cd FinSight
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API Key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit the `.env` file to GitHub.

### 5. Run the Application

```bash
python -m streamlit run app.py
```

The Streamlit application will open in your browser.

---

## Configuration

The project configuration is stored in `config.yaml`.

Current configuration:

```yaml
llm:
  provider: groq
  model: openai/gpt-oss-20b
  temperature: 0.2

pipeline:
  confidence_threshold: 0.5
```

The LLM temperature controls the randomness of the generated response.

The confidence threshold controls when the ML classifier should fall back to the LLM.

---

## Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Linear SVM
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

### NLP

- TF-IDF
- Text preprocessing
- BANKING77

### Generative AI

- Groq API
- GPT-OSS-20B
- Structured JSON output
- Prompt engineering

### Validation

- Pydantic

### Application

- Streamlit

### Data Processing

- Pandas
- NumPy

### Model Persistence

- Joblib

---

## Security

API credentials are stored in environment variables using a `.env` file.

The API key is not hardcoded inside the source code.

The `.env` file should never be committed to the GitHub repository.

---

## Limitations

The current implementation has several limitations:

- The domain guard uses keyword matching and can produce false positives or false negatives.
- SVM decision margins are not calibrated probabilities.
- The LLM depends on an external API.
- API rate limits can interrupt LLM requests.
- LLM responses depend on the quality of the prompt and model behavior.
- The spelling checker is intentionally lightweight.
- The hybrid experiment was performed on a balanced sample of 154 test queries rather than the complete test set.
- The routing threshold of 0.5 is specific to the current experiment and model configuration.

---

## Future Scope

Potential improvements include:

- Calibrated confidence estimation
- More robust semantic domain detection
- Better fallback handling for API failures
- Response caching
- Expanded evaluation on the complete test set
- More detailed support analytics
- Conversation history
- Human-agent escalation workflows
- Monitoring of ML and LLM routing performance
- Deployment using a production cloud environment

---

## Project Outcome

FinSight demonstrates how traditional Machine Learning and Generative AI can be combined into a single customer-support workflow.

The Linear SVM provides fast intent classification for straightforward queries, while the LLM provides deeper analysis for lower-confidence cases.

The project combines:

```text
NLP
+
Machine Learning
+
Confidence-Based Routing
+
Generative AI
+
Structured Output Validation
+
Streamlit
```

This creates an end-to-end customer-support intelligence system rather than a standalone text-classification model.

---

## Author

**Aaryan Bhatia**

B.Tech Data Science  
Manipal University Jaipur

---

## Disclaimer

This project is developed for educational and portfolio purposes using the BANKING77 dataset. It is not intended to provide real financial advice or operate as a production banking support system.
