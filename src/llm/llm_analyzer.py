import json
from pathlib import Path

import joblib
import yaml
from dotenv import load_dotenv
from groq import Groq

from src.schemas.support_schema import SupportAnalysis


BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

model_name = config["llm"]["model"]
temperature = float(config["llm"]["temperature"])

svm_model = joblib.load(
    BASE_DIR / "models" / "svm_model.pkl"
)

allowed_intents = list(svm_model.classes_)

client = Groq()

with open(
    BASE_DIR / "prompts" / "support_analysis.txt",
    "r",
    encoding="utf-8"
) as file:
    prompt_template = file.read()


def analyze_with_llm(customer_query):

    intent_list = "\n".join(
        f"- {intent}" for intent in allowed_intents
    )

    prompt = prompt_template.format(
        customer_query=customer_query
    )

    system_prompt = f"""
You are SupportIQ, an intelligent banking customer-support analysis system.

You MUST return ONLY valid JSON.
Do not use Markdown.
Do not include explanations outside the JSON.

The JSON must contain exactly these fields:
- intent
- sentiment
- urgency
- summary
- entities
- root_cause
- recommended_action
- department
- escalation_required
- customer_response

The intent field MUST be exactly one of the following BANKING77 intents:

{intent_list}

Do not create a new intent.
Do not rename an intent.
Do not combine multiple intents.
"""

    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    llm_output = response.choices[0].message.content

    parsed_output = json.loads(llm_output)

    validated_output = SupportAnalysis(**parsed_output)

    if validated_output.intent not in allowed_intents:
        raise ValueError(
            "LLM returned an intent outside the BANKING77 intent set."
        )

    return validated_output