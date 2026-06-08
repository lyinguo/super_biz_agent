from openai import OpenAI

# 1. 填入你的新 Key
api_key = "sk-ctqzhwwosissprjawdlpqnrveejmozxissegxggwiklixrpu" 

# 2. 硅基流动的 Base URL
base_url = "https://api.siliconflow.cn/v1"

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    response = client.chat.completions.create(
        model="MiniMaxAI/MiniMax-M2.5", # 确保这是硅基流动支持的模型名称
        messages=[{"role": "user", "content": "你好"}]
    )
    print("✅ 连接成功！模型回复:", response.choices[0].message.content)
except Exception as e:
    print(f"❌ 依然报错: {e}")