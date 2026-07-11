import gradio as gr
import os
from openai import OpenAI

GROK_API_KEY = os.getenv("GROK_API_KEY")

SYSTEM_PROMPT = """你是 DealForge Agent，一個專業、務實的 Multi-Deal Discount 投資策略 AI Assistant。

你的目標是幫助用戶在 12 個月內透過多個有折扣的 deal 實現利潤目標。

回應原則：
- 即使用戶資料不完整，你都要盡量用合理假設先計算結果，並清楚說明你用了哪些假設。
- 不要一直追問用戶補充資料，除非真的無法計算。
- 回覆要有結構：用列表、重點分明，必要時分 Base Case / Optimistic / Pessimistic 三種情景。
- 計算要盡量精準，解釋要清晰直接。
- 如果出現錯誤或無法計算，要誠實告知用戶，並建議解決方法。
- 用中文回應，語言專業但易明。"""

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
            model="grok-4.5",
            messages=messages,
            temperature=0.7,
            max_tokens=1800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"系統出現錯誤：{str(e)}\n請稍後再試或檢查輸入資料。"

demo = gr.ChatInterface(
    respond,
    title="DealForge",
    description="幫助你透過多個折扣 deal 優化 12 個月投資策略的 AI 工具"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
