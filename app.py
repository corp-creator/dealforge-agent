import gradio as gr
import os
import json
from openai import OpenAI
from typing import List, Dict
import tempfile

GROK_API_KEY = os.getenv("GROK_API_KEY")

# ==================== Tool ====================
def calculate_portfolio(deals: List[Dict]) -> str:
    total_allocation = 0
    total_expected_profit = 0

    for deal in deals:
        allocation = deal.get("allocation", 0)
        discount = deal.get("discount", 0)
        expected_gain = deal.get("expected_gain", 0)
        success_rate = deal.get("success_rate", 0)
        trim_ratio = deal.get("trim_ratio", 0.6)

        effective_entry = allocation * (1 + discount)
        expected_exit = effective_entry * (1 + expected_gain)
        gross_profit = expected_exit * trim_ratio - allocation * trim_ratio
        net_profit = gross_profit * (1 - 0.23) * success_rate

        total_allocation += allocation
        total_expected_profit += net_profit

    return json.dumps({
        "total_allocation": round(total_allocation, 0),
        "total_expected_profit": round(total_expected_profit, 0),
        "expected_roi": round((total_expected_profit / total_allocation) * 100, 1) if total_allocation > 0 else 0
    }, ensure_ascii=False)


SYSTEM_PROMPT = """你是 DealForge Agent，一個專業務實的投資策略 AI Assistant。

你的目標是幫助用戶透過多個折扣 deal 在 12 個月內實現利潤目標。

回應原則：
- 資料不完整時，用合理假設先計算，並清楚說明假設。
- 不要一直追問，用戶體驗優先。
- 回覆要有結構（列表 + 分情景）。
- 用中文，專業但易明。"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio",
            "description": "計算多個 deal 的總 allocation、預期利潤同 ROI",
            "parameters": {
                "type": "object",
                "properties": {
                    "deals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "allocation": {"type": "number"},
                                "discount": {"type": "number"},
                                "expected_gain": {"type": "number"},
                                "success_rate": {"type": "number"},
                                "trim_ratio": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["deals"]
            }
        }
    }
]

def respond(message, history, session_state):
    if not GROK_API_KEY:
        return "請設定 GROK_API_KEY", history, session_state

    client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="grok-4.5",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1800
        )

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "calculate_portfolio":
                args = json.loads(tool_call.function.arguments)
                result = calculate_portfolio(args["deals"])
                messages.append({"role": "assistant", "content": None, "tool_calls": response.choices[0].message.tool_calls})
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

                final = client.chat.completions.create(model="grok-4.5", messages=messages, temperature=0.7)
                reply = final.choices[0].message.content
        else:
            reply = response.choices[0].message.content

        history.append((message, reply))
        return "", history, session_state

    except Exception as e:
        return f"出錯：{str(e)}", history, session_state


def save_session(history):
    if not history:
        return None
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp.name, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return temp.name

def load_session(file):
    if file is None:
        return []
    with open(file.name, "r", encoding="utf-8") as f:
        return json.load(f)


with gr.Blocks(title="DealForge") as demo:
    gr.Markdown("# DealForge\n幫助你透過多個折扣 deal 優化 12 個月投資策略的 AI 工具")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="輸入你的 deal 資料或問題...", scale=4)
    with gr.Row():
        submit = gr.Button("發送", variant="primary")
        clear = gr.Button("清除對話")
        save_btn = gr.Button("儲存對話")
        load_btn = gr.UploadButton("載入對話", file_types=[".json"])

    session_state = gr.State([])

    submit.click(respond, [msg, chatbot, session_state], [msg, chatbot, session_state])
    msg.submit(respond, [msg, chatbot, session_state], [msg, chatbot, session_state])
    clear.click(lambda: ([], []), None, [chatbot, session_state])
    save_btn.click(save_session, chatbot, gr.File())
    load_btn.upload(load_session, load_btn, chatbot)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
