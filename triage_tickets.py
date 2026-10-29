
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import mock_llm


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
CUSTOMERS_FILE = ROOT / "data" / "raw" / "customers.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

CATEGORIES = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05
MAX_RETRIES = 3
LONG_TICKET_LIMIT = 400
TOKEN_BUDGET = 100000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. PII redaction
# Redact email, phone, and order ID in this order.
# --------------------------------------------------

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")


def redact_text(text):
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = ORDER_PATTERN.sub("[ORDER_ID]", text)
    return text


def count_matches(pattern, text):
    return len(pattern.findall(text))


# --------------------------------------------------
# 3. Classification prompts
# --------------------------------------------------

ZERO_SHOT_PROMPT = """
CLASSIFY
You classify Instamart customer support tickets.
Choose exactly one category from: billing, delivery, technical, fraud.
Return JSON only, in this format:
{"category":"billing","reason":"short explanation"}
Do not add markdown or any text outside the JSON.
""".strip()


FEW_SHOT_PROMPT = """
CLASSIFY
You classify Instamart customer support tickets.
Choose exactly one category from: billing, delivery, technical, fraud.
Return JSON only, in this format:
{"category":"billing","reason":"short explanation"}
Do not add markdown or any text outside the JSON.

EXAMPLES:
Ticket: I was charged twice for my purchase.
Output: {"category":"billing","reason":"duplicate charge"}

Ticket: Someone hacked my account and stole my wallet balance.
Output: {"category":"fraud","reason":"unauthorized account activity"}

Ticket: My delivery is late and the rider has not arrived.
Output: {"category":"delivery","reason":"late delivery"}

Ticket: The app crashes whenever I try to log in.
Output: {"category":"technical","reason":"app login problem"}
""".strip()


# --------------------------------------------------
# 4. Guardrails and model call with retries
# --------------------------------------------------

def call_with_retries(messages, ticket_id):
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise

            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={retry_number} "
                f"wait_seconds={wait_seconds:.4f}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Retry loop ended unexpectedly")


def classify_output(response):
    content = response.choices[0].message.content or ""

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        words = re.findall(r"[a-z]+", content.lower())
        found = [word for word in words if word in CATEGORIES]

        if len(found) == 1:
            return found[0], "classified_keyword"

        return "", "flagged_invalid_output"

    # Valid JSON with an invalid category must be flagged.
    category = parsed.get("category")

    if isinstance(category, str) and category in CATEGORIES:
        return category, "classified_json"

    return "", "flagged_invalid_output"


def process_ticket(row, system_prompt, budget, running_tokens):
    ticket_id = row["ticket_id"]
    original_text = "" if pd.isna(row["text"]) else str(row["text"])

    result = {
        "ticket_id": ticket_id,
        "customer_id": row["customer_id"],
        "true_category": row["true_category"],
        "status": "",
        "category": "",
        "tokens_used": 0,
        "segment_name": "",
        "priority": "NORMAL",
        "route": "human_review",
    }

    if not original_text.strip():
        result["status"] = "skipped_empty"
        return result, running_tokens

    estimated_tokens = mock_llm.est_tokens(original_text)

    if estimated_tokens > LONG_TICKET_LIMIT:
        result["status"] = "flagged_long"
        return result, running_tokens

    # Budget is checked before every model call.
    if running_tokens >= budget:
        result["status"] = "budget_exhausted"
        return result, running_tokens

    safe_text = redact_text(original_text)

    # Safety assertion: none of the original PII patterns
    # should remain in the text sent to the model.
    assert not EMAIL_PATTERN.search(safe_text), (
        f"Email remains after redaction in {ticket_id}"
    )
    assert not PHONE_PATTERN.search(safe_text), (
        f"Phone number remains after redaction in {ticket_id}"
    )
    assert not ORDER_PATTERN.search(safe_text), (
        f"Order ID remains after redaction in {ticket_id}"
    )

    # The user message is exactly the redacted ticket text.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": safe_text},
    ]

    response = call_with_retries(messages, ticket_id)
    tokens_used = response.usage.total_tokens
    running_tokens += tokens_used

    category, status = classify_output(response)

    result["status"] = status
    result["category"] = category
    result["tokens_used"] = tokens_used

    return result, running_tokens


# --------------------------------------------------
# 5. Run a complete classification experiment
# --------------------------------------------------

def run_experiment(tickets, system_prompt, budget, run_name):
    mock_llm.reset()
    running_tokens = 0
    results = []

    print(f"\n{'=' * 55}")
    print(f"Running: {run_name}")
    print(f"Token budget: {budget}")
    print(f"{'=' * 55}")

    for _, row in tickets.iterrows():
        result, running_tokens = process_ticket(
            row=row,
            system_prompt=system_prompt,
            budget=budget,
            running_tokens=running_tokens,
        )
        results.append(result)

    result_df = pd.DataFrame(results)

    classified = result_df[
        result_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    correct = (
        classified["category"].str.lower()
        == classified["true_category"].str.lower()
    )

    accuracy = correct.mean() * 100 if len(classified) else 0

    print(f"\nResults: {run_name}")
    print(f"Total tickets: {len(result_df)}")
    print(f"Classified tickets: {len(classified)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total tokens: {running_tokens}")
    print(
        "Tokens per classified ticket: "
        f"{running_tokens / len(classified):.2f}"
        if len(classified)
        else "Tokens per classified ticket: N/A"
    )
    print("\nStatus counts:")
    print(result_df["status"].value_counts().to_string())

    return result_df, running_tokens


# --------------------------------------------------
# 6. Load data and audit PII redaction
# --------------------------------------------------

def main():
    tickets = pd.read_csv(TICKETS_FILE, keep_default_na=False)
    print(f"Loaded {len(tickets)} tickets from {TICKETS_FILE}")

    email_count = 0
    phone_count = 0
    order_count = 0

    for text in tickets["text"].astype(str):
        email_count += count_matches(EMAIL_PATTERN, text)
        phone_count += count_matches(PHONE_PATTERN, text)
        order_count += count_matches(ORDER_PATTERN, text)

    print("\nPII found before redaction:")
    print(f"Emails: {email_count}")
    print(f"Phone numbers: {phone_count}")
    print(f"Order IDs: {order_count}")
    print(f"Total PII matches: {email_count + phone_count + order_count}")

    # --------------------------------------------------
    # 7. Zero-shot and few-shot comparison
    # --------------------------------------------------

    zero_results, zero_tokens = run_experiment(
        tickets,
        ZERO_SHOT_PROMPT,
        budget=100000,
        run_name="Zero-shot",
    )

    few_results, few_tokens = run_experiment(
        tickets,
        FEW_SHOT_PROMPT,
        budget=100000,
        run_name="Few-shot",
    )

    zero_results.to_csv(
        OUTPUT_DIR / "triage_zero_shot.csv",
        index=False,
    )
    few_results.to_csv(
        OUTPUT_DIR / "triage_few_shot.csv",
        index=False,
    )

    # Confusion table for zero-shot predictions.
    zero_classified = zero_results[
        zero_results["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    print("\nZero-shot confusion table:")
    if len(zero_classified):
        print(
            pd.crosstab(
                zero_classified["true_category"],
                zero_classified["category"],
                rownames=["True category"],
                colnames=["Predicted category"],
                margins=True,
            ).to_string()
        )

    # --------------------------------------------------
    # 8. Zero-shot token-budget demonstration
    # --------------------------------------------------

    budget_results, budget_tokens = run_experiment(
        tickets,
        ZERO_SHOT_PROMPT,
        budget=6000,
        run_name="Zero-shot budget demo",
    )

    if not (budget_results["status"] == "budget_exhausted").any():
        print("No ticket was blocked at 6000 tokens; trying 5000.")
        budget_results, budget_tokens = run_experiment(
            tickets,
            ZERO_SHOT_PROMPT,
            budget=5000,
            run_name="Zero-shot budget demo (5000)",
        )

    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    budget_results.to_csv(budget_path, index=False)
    print(f"\nBudget demo saved to {budget_path}")

    # --------------------------------------------------
    # 9. Final few-shot triage and routing
    # --------------------------------------------------

    final_results, final_tokens = run_experiment(
        tickets,
        FEW_SHOT_PROMPT,
        budget=TOKEN_BUDGET,
        run_name="Final few-shot triage",
    )

    customers = pd.read_csv(CUSTOMERS_FILE, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS_FILE, keep_default_na=False)

    # Treat a customer as premium only when the segment label says so.
    premium_ids = set(
        segments.loc[
            segments["segment_name"].astype(str).str.strip().str.lower()
            == "premium",
            "customer_id",
        ].astype(str)
    )

    final_results["customer_id"] = final_results["customer_id"].astype(str)

    # Keep segment_name blank when no segment is found.
    segment_map = dict(
        zip(
            segments["customer_id"].astype(str),
            segments["segment_name"].astype(str),
        )
    )

    final_results["segment_name"] = (
        final_results["customer_id"].map(segment_map).fillna("")
    )

    for index, result in final_results.iterrows():
        status = result["status"]
        category = result["category"]
        customer_id = str(result["customer_id"])

        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            route = "human_review"
            priority = "NORMAL"

        elif category == "fraud":
            route = "security_team"
            priority = "HIGH"

        elif customer_id in premium_ids:
            route = "premium_desk"
            priority = "HIGH"

        else:
            route = "support_queue"
            priority = "NORMAL"

        final_results.at[index, "route"] = route
        final_results.at[index, "priority"] = priority

    final_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    final_path = OUTPUT_DIR / "triage_results.csv"
    final_results[final_columns].to_csv(final_path, index=False)

    print(f"\nFinal triage file saved to {final_path}")
    print("\nFinal routing counts:")
    print(final_results["route"].value_counts().to_string())

    print("\nFinal status counts:")
    print(final_results["status"].value_counts().to_string())

    print("\nPart C processing finished.")


if __name__ == "__main__":
    main()
