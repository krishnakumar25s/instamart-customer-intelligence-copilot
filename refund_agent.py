
import json
import sqlite3
from pathlib import Path

import mock_llm
from policy_rag import retrieve


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 10
AUTO_REFUND_LIMIT = 1000.00


# Tool definitions for the mock LLM.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order in the SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve relevant policy passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund eligibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {
                        "type": ["number", "null"]
                    },
                },
                "required": ["order_amount", "delivery_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Simulate issuing an eligible refund up to INR 1000.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate refunds above INR 1000 for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
]


# ============================================================
# 2. DATABASE FUNCTIONS
# ============================================================

def get_table_names(connection):
    """Return table names from SQLite."""
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    return {row[0] for row in rows}


def get_order(order_id):
    """Retrieve order details from the cleaned orders table."""
    if not isinstance(order_id, str):
        return {"error": "Invalid order ID"}

    if not order_id.startswith("ORD-"):
        return {"error": "Invalid order ID"}

    if not DB_FILE.exists():
        return {"error": f"Database not found: {DB_FILE}"}

    try:
        with sqlite3.connect(DB_FILE) as connection:
            connection.row_factory = sqlite3.Row

            tables = get_table_names(connection)

            if "orders" not in tables:
                return {
                    "error": "Orders table not found",
                    "available_tables": sorted(tables),
                }

            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(orders)"
                )
            }

            required = {
                "order_id",
                "status",
                "order_amount",
                "delivery_minutes",
            }

            if not required.issubset(columns):
                return {
                    "error": "Required columns missing",
                    "available_columns": sorted(columns),
                }

            row = connection.execute(
                """
                SELECT order_id, status, order_amount, delivery_minutes
                FROM orders
                WHERE order_id = ?
                LIMIT 1
                """,
                (order_id,),
            ).fetchone()

            if row is None:
                return {
                    "error": "Order not found",
                    "order_id": order_id,
                }

            order = dict(row)

            # Normalize values before passing them to the mock model.
            order["status"] = str(order["status"]).strip().lower()

            if order["order_amount"] is not None:
                order["order_amount"] = float(order["order_amount"])

            if order["delivery_minutes"] is not None:
                order["delivery_minutes"] = float(
                    order["delivery_minutes"]
                )

            return order

    except sqlite3.Error as exc:
        return {"error": f"Database error: {exc}"}


# ============================================================
# 3. POLICY RETRIEVAL
# ============================================================

def search_policy(query):
    """Retrieve the three most relevant policy chunks."""
    results = retrieve(query, k=3)

    return {
        "query": query,
        "results": results,
    }


# ============================================================
# 4. REFUND CALCULATION AND ACTIONS
# ============================================================

def calculate_refund(order_amount, delivery_minutes):
    """Calculate the refund according to the project rules."""
    if order_amount is None or delivery_minutes is None:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "reason": "Missing order amount or delivery duration; human review required.",
        }

    amount = float(order_amount)
    minutes = float(delivery_minutes)

    if amount < 0:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "reason": "Invalid negative order amount.",
        }

    if minutes <= 45:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "reason": "Delivery was within 45 minutes.",
        }

    return {
        "refund_inr": round(amount, 2),
        "eligible": True,
        "reason": "Completed order delivered after 45 minutes.",
    }


def issue_refund(order_id, refund_inr):
    """Simulate issuing an eligible refund."""
    amount = round(float(refund_inr), 2)

    if amount <= 0:
        return {"error": "Refund must be greater than zero."}

    if amount > AUTO_REFUND_LIMIT:
        return {
            "error": "Refunds above INR 1000 require human approval."
        }

    # This is a simulated action; no real money is moved.
    return {
        "status": "issued",
        "order_id": order_id,
        "refund_inr": amount,
        "message": "Simulated refund issued successfully.",
    }


def escalate_to_human(order_id, refund_inr):
    """Simulate escalating a large refund for human approval."""
    amount = round(float(refund_inr), 2)

    if amount <= AUTO_REFUND_LIMIT:
        return {
            "error": "Only refunds above INR 1000 should be escalated."
        }

    return {
        "status": "escalated",
        "order_id": order_id,
        "refund_inr": amount,
        "message": "Sent to a human support lead for approval.",
    }


# ============================================================
# 5. TOOL EXECUTION
# ============================================================

def execute_tool(name, arguments):
    """Execute a named tool with its arguments."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name == "get_order":
        return get_order(**arguments)

    if name == "search_policy":
        return search_policy(**arguments)

    if name == "calculate_refund":
        return calculate_refund(**arguments)

    if name == "issue_refund":
        return issue_refund(**arguments)

    if name == "escalate_to_human":
        return escalate_to_human(**arguments)

    return {"error": f"Unknown tool: {name}"}


# ============================================================
# 6. AGENT WITH LOOP DETECTION AND REFUND SAFEGUARDS
# ============================================================

def run_agent(customer_message, max_steps=MAX_STEPS):
    """Run the mock LLM agent while enforcing refund safety rules."""
    mock_llm.reset()

    messages = [
        {
            "role": "system",
            "content": (
                "AGENT\n"
                "You are an Instamart refund support agent. "
                "Use get_order to look up the order. "
                "Use search_policy to retrieve relevant policy text. "
                "Only completed orders qualify for refunds. "
                "A completed order delivered after 45 minutes qualifies "
                "for a full refund of its order amount. "
                "A completed order delivered in 45 minutes or less "
                "does not qualify for a late-delivery refund. "
                "Refunds up to INR 1000 may be issued automatically. "
                "Refunds above INR 1000 must be escalated to a human. "
                "Never invent order details or bypass a tool result."
            ),
        },
        {
            "role": "user",
            "content": customer_message,
        },
    ]

    trace = []
    seen_calls = set()
    verified_order = None
    policy_retrieved = False
    refund_calculation = None

    for step in range(max_steps):
        response = mock_llm.chat(messages, tools=TOOLS)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return {
                "answer": message.content or "",
                "trace": trace,
                "steps": step + 1,
            }

        assistant_tool_calls = []

        for call in tool_calls:
            arguments = call.arguments

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return {
                        "answer": "Stopped: invalid tool arguments.",
                        "trace": trace,
                        "steps": step + 1,
                    }

            signature = (
                call.name,
                json.dumps(arguments, sort_keys=True, default=str),
            )

            if signature in seen_calls:
                trace.append({
                    "step": step + 1,
                    "tool": call.name,
                    "arguments": arguments,
                    "result": {"error": "Repeated tool call detected."},
                })

                return {
                    "answer": "Stopped: repeated tool call detected.",
                    "trace": trace,
                    "steps": step + 1,
                }

            seen_calls.add(signature)

            assistant_tool_calls.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(arguments),
                },
            })

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": assistant_tool_calls,
        })

        for call in tool_calls:
            name = call.name
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            # Always retrieve and verify the order before refund actions.
            if name == "get_order":
                result = execute_tool(name, arguments)

                if "error" not in result:
                    verified_order = result
                    policy_retrieved = False
                    refund_calculation = None

            elif name == "search_policy":
                if verified_order is None:
                    result = {
                        "error": "Look up a valid order before retrieving refund policy."
                    }
                elif verified_order["status"] != "completed":
                    result = {
                        "error": "Only completed orders can be refunded."
                    }
                else:
                    result = execute_tool(name, arguments)
                    policy_retrieved = True

            elif name == "calculate_refund":
                if verified_order is None:
                    result = {"error": "No verified order found."}

                elif verified_order["status"] != "completed":
                    result = {
                        "error": "Only completed orders can be refunded."
                    }

                elif not policy_retrieved:
                    result = {
                        "error": "Retrieve the refund policy before calculating."
                    }

                else:
                    # Use the verified database values, not model-supplied
                    # values, to calculate the refund.
                    result = calculate_refund(
                        order_amount=verified_order["order_amount"],
                        delivery_minutes=verified_order["delivery_minutes"],
                    )

                    refund_calculation = result

            elif name in {"issue_refund", "escalate_to_human"}:
                order_id = arguments.get("order_id")
                requested_amount = arguments.get("refund_inr")

                if verified_order is None:
                    result = {"error": "No verified order found."}

                elif verified_order["status"] != "completed":
                    result = {
                        "error": "Only completed orders can be refunded."
                    }

                elif not policy_retrieved or refund_calculation is None:
                    result = {
                        "error": "Refund policy and calculation are required first."
                    }

                elif order_id != verified_order["order_id"]:
                    result = {
                        "error": "The requested order ID does not match the verified order."
                    }

                elif not refund_calculation.get("eligible"):
                    result = {
                        "error": refund_calculation["reason"]
                    }

                elif round(float(requested_amount), 2) != round(
                    float(refund_calculation["refund_inr"]), 2
                ):
                    result = {
                        "error": "Requested refund does not match the calculated amount."
                    }

                elif (
                    name == "issue_refund"
                    and float(requested_amount) > AUTO_REFUND_LIMIT
                ):
                    result = {
                        "error": "Refunds above INR 1000 must be escalated."
                    }

                elif (
                    name == "escalate_to_human"
                    and float(requested_amount) <= AUTO_REFUND_LIMIT
                ):
                    result = {
                        "error": "Only refunds above INR 1000 should be escalated."
                    }

                else:
                    result = execute_tool(name, arguments)

            else:
                result = execute_tool(name, arguments)

            trace.append({
                "step": step + 1,
                "tool": name,
                "arguments": arguments,
                "result": result,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result, default=str),
            })

    return {
        "answer": "Stopped: maximum tool steps reached.",
        "trace": trace,
        "steps": max_steps,
    }


def find_late_delivery_orders():
    """Find completed orders delivered after 45 minutes."""
    import sqlite3

    db_path = ROOT / "data" / "instamart.db"

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE LOWER(status) = 'completed'
              AND delivery_minutes > 45
            ORDER BY order_amount
            LIMIT 10
            """
        ).fetchall()

    print("\nCompleted orders delivered after 45 minutes:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    find_late_delivery_orders()

# ============================================================
# 7. TEST SCENARIOS
# ============================================================

if __name__ == "__main__":
    scenarios = [
        "Please check my refund. Order ID ORD-100001",
        "I want a refund for order ORD-999999",
        "Please check this order: ORD-100002",
        "I need a refund but I do not know my order ID.",
        "Please check my refund for order ORD-107374",
        "Please check my refund for order ORD-102006",
        "Please check my refund for ORD-100001 [LOOP]","Please check my refund for order ORD-107652",
    ]

    for customer_message in scenarios:
        print("\n" + "=" * 65)
        print("CUSTOMER:", customer_message)

        result = run_agent(customer_message)

        print("AGENT:", result["answer"])
        print("TOOL TRACE:")

        for item in result["trace"]:
            print(
                f"  Step {item['step']}: {item['tool']} "
                f"args={item['arguments']} "
                f"result={item['result']}"
            )

        print("Steps:", result["steps"])

    print("\nAll demo scenarios finished.")
