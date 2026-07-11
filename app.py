import gradio as gr
import os
from openai import OpenAI

GROK_API_KEY = os.getenv("GROK_API_KEY")

SYSTEM_PROMPT = """你是 DealForge Agent，一個專門幫助用戶做 Multi-Deal Discount 投資策略的 AI Agent。

你的任務是幫助用戶透過多個有折扣的 deal，在 12 個月內實現較高利潤。

請用自然語言與用戶溝通，幫助他們輸入 deal 資料、計算預期回報、做情景分析，並給出合理建議。"""

def respond(message, history):
    if not GROK_API_KEY:
        return "請喺部署平台設定 GROK_API_KEY Secret。"
    
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
            model="grok-2-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"出錯：{str(e)}"

# 重要：加咗 server_name 同 server_port
demo = gr.ChatInterface(
    respond,
    title="📈 DealForge Agent",
    description="用 Grok API 實現嘅 Deal 策略 AI Assistant"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
