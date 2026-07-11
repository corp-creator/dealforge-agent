import gradio as gr
import os
import json
from openai import OpenAI
from typing import List, Dict

GROK_API_KEY = os.getenv("GROK_API_KEY")

# ==================== Tool: 計算多個 Deal 回報 ====================
def calculate_portfolio(deals: List[Dict]) -> str:
    """
    計算多個 deal 的預期回報
    deals 格式: [{"allocation": 40000, "discount": 0.28, "expected_gain": 0.80, "success_rate": 0.75, "trim_ratio": 0.60}, ...]
    """
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
        net_profit = gross_profit * (1 - 0.23) * success_rate   # 扣 23% drag

        total_allocation += allocation
        total_expected_profit += net_profit

    return json.dumps({
        "total_allocation": round(total_allocation, 0),
        "total_expected_profit": round(total_expected_profit, 0),
        "expected_roi": round((total_expected_profit / total_allocation) * 100, 1) if total_allocation > 0 else 0
    }, ensure_ascii=False)


# ==================== System Prompt ====================
SYSTEM_PROMPT = """你是 DealForge Agent，一個專業、務實的 Multi-Deal Discount 投資策略 AI Assistant。

你的目標是幫助用戶在 12 個月內透過多個有折扣的 deal 實現利潤目標。

回應原則：
- 即使用戶資料不完整，你都要盡量用合理假設先計算結果，並清楚說明你用了哪些假設。
- 不要一直追問用戶補充資料，除非真的無法計算。
- 回覆要有結構：用列表、重點分明，必要時分 Base Case / Optimistic / Pessimistic 三種情景。
- 計算要盡量精準，解釋要清晰直接。
- 如果出現錯誤或無法計算，要誠實告知用戶，並建議解決方法。
- 用中文回應，語言專業但易明。"""

# ==================== Tool 定义 ====================
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
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1800
        )
        
        # 處理 tool call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "calculate_portfolio":
                result = calculate_portfolio(arguments["deals"])
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": response.choices[0].message.tool_calls
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                
                final_response = client.chat.completions.create(
                    model="grok-4.5",
                    messages=messages,
                    temperature=0.7
                )
                return final_response.choices[0].message.content
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"系統出現錯誤：{str(e)}\n請稍後再試。"


demo = gr.ChatInterface(
    respond,
    title="DealForge",
    description="幫助你透過多個折扣 deal 優化 12 個月投資策略的 AI 工具"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
