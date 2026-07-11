import gradio as gr
import os
from openai import OpenAI

GROK_API_KEY = os.getenv("GROK_API_KEY")

SYSTEM_PROMPT = """你是 DealForge Agent，一個專業務實的投資策略 AI Assistant。

你的目標是幫助用戶透過多個折扣 deal 在 12 個月內實現利潤目標。

回應原則：
- 資料不完整時，用合理假設先計算，並清楚說明假設。
- 不要一直追問用戶。
- 回覆要有結構（用列表、分情景）。
- 用中文，專業但易明。"""

def respond(message, history):
    if not GROK_API_KEY:
        return "請設定 GROK_API_KEY", history

    client = OpenAI(
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="grok-4.5",
            messages=messages,
            temperature=0.7,
            max_tokens=1800
        )
        reply = response.choices[0].message.content
        history.append((message, reply))
        return "", history
    except Exception as e:
        return f"出錯：{str(e)}", history


with gr.Blocks(title="DealForge") as demo:
    gr.Markdown("# DealForge\n幫助你透過多個折扣 deal 優化 12 個月投資策略的 AI 工具")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="輸入問題或 deal 資料...")
    clear = gr.Button("清除對話")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
