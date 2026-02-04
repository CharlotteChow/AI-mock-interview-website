# AI Oral Interview Practice Agent（Streamlit）

一个开箱即用的口语面试练习应用：上传简历、粘贴 JD、录音回答问题，得到 AI 打分与改进建议。

## 你可以用它做什么

- 上传 1 份或多份简历（`PDF/TXT`），自动合并为提问上下文。
- 粘贴岗位 JD，生成更贴岗的面试问题。
- 每题支持计时（`60s/120s`）、录音、转写和即时反馈。
- 自动给出每题解析（面试官看重点、关键要点、回答框架、关键词）。
- 题后生成结构化反馈：`Score / Strengths / Weaknesses / Refined Sample Answer / Perfect Answer（基于 JD+Resume）`。
- Perfect Answer 会按题目时长控制篇幅（正常语速）：`60s` 约 `100+` 词，`120s` 约 `200+` 词。
- Section 结束后生成总评与改进方向。
- 支持上下文记忆（跨题/跨 section）和本地保存设置。

## 快速开始（3 分钟）

### 1) 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 启动应用

```bash
streamlit run app.py
```

启动后浏览器会打开本地页面（通常是 `http://localhost:8501`）。

### 3) 第一次使用流程

1. 在左侧填写 `OpenAI API Key`（可留空）。
2. （可选）填写 `API Base URL` 用于兼容 OpenAI 协议的服务。
3. 上传简历、粘贴 JD。
4. 选择每个 section 的题量、时长、模型，点击 **Start Interview Section**。
5. 每题点击 **Start Answer** 开始计时，录音后点击 **Submit & Analyze**。
6. 查看反馈后点 **Next Question**，最后进入 **Section Review** 总结页（包含 transcript 改写版 + 按时长生成的完美回答）。

### Third-Party API 快速接入

- 左侧点击 **Use Third-Party API**，会自动填入：
  - Base URL: `https://co.yes.vg/v1/responses`
  - Model: `gpt-5.2-codex`
- 再填入你的 YesCode Key 即可（也支持环境变量 `YESCODE_API_KEY`）。

## 两种运行模式

### Online 模式（有 API Key）

- 真实调用模型生成问题、题目解析、反馈和 section 总结。

### Mock 模式（无 API Key）

- 不调用 API，也可以完整体验 UI 流程（问题/反馈为模拟内容）。
- 适合先调流程、测界面、试录音。

## 录音与转写机制（自动兜底）

- 录音优先使用 `streamlit-audiorecorder`；若不可用则降级到 Streamlit 内置录音。
- 转写优先调用提供商语音转写接口；失败时自动回退到本地 `faster-whisper`。
- 侧边栏可选本地转写模型：`base`（默认，更快）/ `small`（速度与精度平衡）/ `medium`（更准但更慢）。
- 若你希望 `streamlit-audiorecorder` 正常工作，请确保本机有 `ffmpeg/ffprobe`（macOS 可用 `brew install ffmpeg`）。

## 可选环境变量

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-provider.com/v1"
export OPENAI_MODEL="gpt-4o-mini"

# YesCode（可二选一或混用 OpenAI 变量）
export YESCODE_API_KEY="..."
export YESCODE_BASE_URL="https://co.yes.vg/v1/responses"
export YESCODE_MODEL="gpt-5.2-codex"

# 本地 faster-whisper 参数（可选）
export LOCAL_WHISPER_MODEL="base"
export LOCAL_WHISPER_DEVICE="auto"
export LOCAL_WHISPER_COMPUTE_TYPE="int8"
```

说明：
- `OPENAI_BASE_URL` 适用于 OpenAI 兼容服务。
- 未设置 `OPENAI_*` 时，会自动读取 `YESCODE_*` 变量作为默认值。
- Base URL 不会再自动补 `/v1`，请按你的服务商文档填写完整地址。
- 使用第三方 Base URL 时，应用会自动把同一个 Key 也作为 `X-API-Key` 请求头发送。

## 本地持久化文件

- `.sidebar_settings.json`：保存侧边栏配置（题量、时长、模型、Base URL、可选 JD 文本）。
- `.context_memory.json`：保存上下文记忆（summary + 历史条目），可在侧边栏下载/导入 JSON。

## 常见问题

- 提示 `403 Invalid client`：通常是 API Key、Base URL 或模型配置不匹配。
- 若同一组凭证持续返回 `403 Invalid client`，应用会自动对该凭证切到 Mock 模式，避免每题重复报错。
- 录音可用但转写失败：先检查 `faster-whisper` 是否安装、`ffmpeg` 是否在 PATH 中。
- PDF 简历读取失败：检查 `pypdf` 是否安装、文件是否损坏或加密。
