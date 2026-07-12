import gradio as gr
import os
from openai import OpenAI

GROK_API_KEY = os.getenv("GROK_API_KEY")

def respond(message, history):
    if not GROK_API_KEY:
        return "請設定 GROK_API_KEY", history

    client = OpenAI(
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1"
    )

    messages = [{"role": "system", "content": "你係 DealForge Agent，一個幫助用戶做投資策略嘅助手。"}]
    
    for msg in history:
        messages.append(msg)

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="grok-4.5",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        reply = response.choices[0].message.content

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})

        return "", history

    except Exception as e:
        return f"出錯：{str(e)}", history


with gr.Blocks(title="DealForge") as demo:
    gr.Markdown("# DealForge")
    chatbot = gr.Chatbot(height=500)          # 已經移除 type="messages"
    msg = gr.Textbox(placeholder="輸入問題...")
    clear = gr.Button("清除對話")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
