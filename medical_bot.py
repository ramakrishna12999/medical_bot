# ============================================================
#   MedAssist AI — Medical Chatbot (Gemini API Edition)
#   Google Colab / Jupyter — Run each cell in order ↓
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────
!pip install -q google-genai

# ─────────────────────────────────────────────────────────────
# CELL 2 — Imports & Configuration
# ─────────────────────────────────────────────────────────────
import os
import json
import time
import datetime
import textwrap
import re
from google import genai
from google.genai import types
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from google.colab import userdata

# ──────────────────────────────────
#  CONFIG  (edit these if needed)
# ──────────────────────────────────
GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")  # Add this key in Colab Secrets panel
MODEL          = "gemini-2.5-flash"               # or "gemini-2.0-flash" for speed
MAX_TOKENS     = 1024
TEMPERATURE    = 0.4

SYSTEM_PROMPT = """
You are MedAssist AI, a knowledgeable and empathetic medical information assistant.

CORE RESPONSIBILITIES:
1. Answer general medical and health questions clearly and accurately.
2. Explain medical conditions, symptoms, and terminology in plain language.
3. Provide information about medications — uses, side effects, and precautions.
4. Offer wellness and preventive care advice.
5. Guide users on WHEN and WHERE to seek professional help.

SAFETY RULES (MUST FOLLOW):
- NEVER diagnose. Use: "This may suggest…", "Common causes include…"
- NEVER prescribe. Say: "Dosage must be set by your doctor or pharmacist."
- For EMERGENCIES (chest pain, stroke, severe bleeding, suicidal thoughts),
  say: "CALL 911 NOW / GO TO THE NEAREST ER IMMEDIATELY."
- Always clarify you are an AI, not a licensed physician.
- Be compassionate, clear, and non-alarmist.

RESPONSE FORMAT:
- Use Markdown: headers (##), bullet points, **bold** for key terms.
- Structure complex topics: Overview → Details → Recommendations → When to See a Doctor.
- End with a gentle reminder to consult a healthcare professional when relevant.
""".strip()

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "difficulty breathing", "stroke", "unconscious", "unresponsive",
    "severe bleeding", "overdose", "suicide", "kill myself",
    "not breathing", "seizure", "anaphylaxis",
]

print("✅ Configuration loaded.")


# ─────────────────────────────────────────────────────────────
# CELL 3 — Core Classes
# ─────────────────────────────────────────────────────────────

# ── Conversation Memory ───────────────────────────────────────
class ConversationManager:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Prune oldest pairs beyond max_turns
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def to_gemini_history(self) -> list[dict]:
        """
        Convert history to Gemini format.
        Gemini uses 'user' and 'model' roles (not 'assistant').
        Excludes the last message — that's sent as the live prompt.
        """
        gemini_msgs = []
        for msg in self.history[:-1]:  # exclude the latest user message
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_msgs.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        return gemini_msgs

    def latest_user_message(self) -> str:
        """Return the last user message to send as the current prompt."""
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def clear(self):
        self.history.clear()

    def save(self, filename: str = "medassist_session.json"):
        data = {
            "app": "MedAssist AI (Gemini)",
            "model": MODEL,
            "saved_at": datetime.datetime.now().isoformat(),
            "messages": self.history,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filename


# ── LLM Client ────────────────────────────────────────────────
class MedicalLLMClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def chat(self, conversation: "ConversationManager", retries: int = 3) -> str:
        """
        Recreate a Gemini chat session each call, pre-seeding it with
        history so multi-turn context is preserved.
        """
        history   = conversation.to_gemini_history()
        user_text = conversation.latest_user_message()

        for attempt in range(1, retries + 1):
            try:
                chat_session = self.client.chats.create(
                    model=MODEL,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                    ),
                    history=history,
                )
                response = chat_session.send_message(user_text)
                return response.text

            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "rate" in err_str:
                    raise RuntimeError("⚠️ Rate limit hit. Please wait and try again.")
                if "api key" in err_str or "authentication" in err_str or "invalid" in err_str:
                    raise RuntimeError("❌ Invalid API key. Check GEMINI_API_KEY in Colab Secrets.")
                if attempt < retries:
                    time.sleep(2 * attempt)
                    continue
                raise RuntimeError(f"❌ API error after {retries} attempts: {e}")

        raise RuntimeError("❌ Failed after retries.")


# ── Safety Layer ──────────────────────────────────────────────
class SafetyLayer:
    @staticmethod
    def is_emergency(text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in EMERGENCY_KEYWORDS)


# ── Markdown → HTML renderer ──────────────────────────────────
def md_to_html(text: str) -> str:
    """Convert simple Markdown to HTML for display in Jupyter."""
    text = re.sub(r"^### (.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$",  r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$",   r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"^\s*[-•]\s+(.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"(<li>.*?</li>(\n|$))+", lambda m: f"<ul>{m.group(0)}</ul>", text, flags=re.DOTALL)
    text = re.sub(r"\n{2,}", "</p><p>", text)
    text = text.replace("\n", "<br>")
    return f"<p>{text}</p>"


print("✅ Core classes ready.")


# ─────────────────────────────────────────────────────────────
# CELL 4 — UI Builder & Chatbot
# ─────────────────────────────────────────────────────────────

# Initialise singletons
conversation = ConversationManager()
llm          = MedicalLLMClient(api_key=GEMINI_API_KEY)
safety       = SafetyLayer()

# ── Shared CSS ────────────────────────────────────────────────
CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Mono&display=swap');

  .med-container {
    font-family: 'Lora', Georgia, serif;
    max-width: 820px;
    margin: 0 auto;
    background: #f7faf9;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
  }
  .med-header {
    background: linear-gradient(135deg, #0d4f45 0%, #1a7a6a 100%);
    color: white;
    padding: 18px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .med-header h2 { margin:0; font-size:22px; letter-spacing:0.03em; }
  .med-header p  { margin:0; font-size:12px; opacity:0.75; }
  .chat-area {
    background: #eef4f3;
    padding: 16px;
    min-height: 380px;
    max-height: 460px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg { display:flex; align-items:flex-start; gap:10px; animation: fadein 0.3s ease; }
  .msg.user  { flex-direction: row-reverse; }
  .avatar {
    width:36px; height:36px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:18px; flex-shrink:0;
  }
  .avatar.bot  { background: linear-gradient(135deg,#1a5f54,#2a9d8f); box-shadow:0 2px 8px rgba(42,157,143,.35); }
  .avatar.user { background:#d4ece9; border:2px solid #2a9d8f; }
  .bubble {
    max-width: 72%;
    padding: 12px 16px;
    border-radius: 4px 18px 18px 18px;
    font-size: 14.5px;
    line-height: 1.7;
  }
  .bubble.bot  { background:white; color:#1a2e2c; box-shadow:0 2px 10px rgba(0,0,0,.07); border:1px solid #ddecea; }
  .bubble.user { background:linear-gradient(135deg,#1a5f54,#2a9d8f); color:white; border-radius:18px 4px 18px 18px; box-shadow:0 4px 14px rgba(42,157,143,.35); }
  .bubble h2,h3,h4 { margin:6px 0 2px; }
  .bubble ul { padding-left:18px; margin:4px 0; }
  .bubble li { margin:3px 0; }
  .ts { font-size:10px; opacity:0.55; margin-top:6px; text-align:right; font-family:'DM Mono',monospace; }
  .typing { display:flex; gap:5px; padding:10px 14px; align-items:center; }
  .dot { width:8px;height:8px;border-radius:50%;background:#2a9d8f; animation:bounce 1.2s ease infinite; }
  .dot:nth-child(2){animation-delay:.2s}
  .dot:nth-child(3){animation-delay:.4s}
  .disclaimer {
    background:#fff8e1; border-top:2px solid #f0c040;
    padding:10px 16px; font-size:12px; color:#5a4a00; text-align:center;
  }
  @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-7px)} }
  @keyframes fadein { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
</style>
"""

def render_message(role: str, content: str) -> str:
    ts  = datetime.datetime.now().strftime("%H:%M")
    cls = "bot" if role == "assistant" else "user"
    icon = "🩺" if role == "assistant" else "👤"
    body = md_to_html(content) if role == "assistant" else f"<p>{content}</p>"
    return f"""
    <div class="msg {cls}">
      <div class="avatar {cls}">{icon}</div>
      <div class="bubble {cls}">{body}<div class="ts">{ts}</div></div>
    </div>"""

def render_typing() -> str:
    return """<div class="msg bot">
      <div class="avatar bot">🩺</div>
      <div class="bubble bot">
        <div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
      </div></div>"""

def render_emergency() -> str:
    return """<div style="background:#ff4444;color:white;padding:12px 16px;border-radius:10px;margin:4px 0;font-weight:bold;">
      🚨 EMERGENCY DETECTED — CALL 911 NOW or GO TO YOUR NEAREST ER IMMEDIATELY. Do not wait.
    </div>"""


# ── Build the full UI ─────────────────────────────────────────
def build_chatbot_ui():
    # ---- Widgets ----
    chat_output  = widgets.Output()
    user_input   = widgets.Textarea(
        placeholder="Type your health question here… (click Send or press the button)",
        layout=widgets.Layout(width="100%", height="72px"),
    )
    send_btn     = widgets.Button(description="Send ➤",  button_style="success",
                                   layout=widgets.Layout(width="90px"))
    clear_btn    = widgets.Button(description="🗑 Clear",  button_style="warning",
                                   layout=widgets.Layout(width="90px"))
    save_btn     = widgets.Button(description="💾 Save",   button_style="info",
                                   layout=widgets.Layout(width="90px"))
    status_lbl   = widgets.Label(value="")

    # Internal chat log (HTML strings)
    chat_log: list[str] = []

    # ── Render entire chat pane ──
    def refresh_chat(extra_html: str = ""):
        with chat_output:
            clear_output(wait=True)
            all_msgs = "\n".join(chat_log) + extra_html
            display(HTML(f"""
            {CSS}
            <div class="med-container">
              <div class="med-header">
                <span style="font-size:32px">🩺</span>
                <div>
                  <h2>MedAssist AI <span style="font-size:13px;opacity:0.7;font-weight:normal;">· Powered by Gemini</span></h2>
                  <p>General Health Information Assistant · Not a substitute for professional care</p>
                </div>
              </div>
              <div class="chat-area" id="chat-area">{all_msgs}</div>
              <div class="disclaimer">
                ⚠️ For educational purposes only. Always consult a licensed healthcare professional.
                &nbsp;|&nbsp; 🚨 Emergency? <strong>Call 911</strong>
              </div>
            </div>
            <script>
              var el = document.querySelector('.chat-area');
              if(el) el.scrollTop = el.scrollHeight;
            </script>
            """))

    # ── Send message handler ──
    def on_send(b=None):
        text = user_input.value.strip()
        if not text:
            return

        user_input.value  = ""
        send_btn.disabled = True
        status_lbl.value  = "⏳ MedAssist is thinking…"

        # Emergency check
        if safety.is_emergency(text):
            chat_log.append(render_emergency())

        # User bubble
        chat_log.append(render_message("user", text))
        conversation.add("user", text)

        # Show typing indicator
        refresh_chat(render_typing())

        # Call Gemini LLM
        try:
            reply = llm.chat(conversation)
        except RuntimeError as e:
            reply = f"⚠️ **Error:** {e}\n\nPlease check your API key and try again."

        # Assistant bubble
        conversation.add("assistant", reply)
        chat_log.append(render_message("assistant", reply))
        refresh_chat()

        send_btn.disabled = False
        status_lbl.value  = f"✅ Done · {len([m for m in conversation.history if m['role']=='user'])} question(s) asked"

    # ── Clear handler ──
    def on_clear(b):
        conversation.clear()
        chat_log.clear()
        refresh_chat()
        status_lbl.value = "🗑 Chat cleared."

    # ── Save handler ──
    def on_save(b):
        fname = conversation.save()
        status_lbl.value = f"💾 Saved → {fname}"

    # ── Wire up events ──
    send_btn.on_click(on_send)
    clear_btn.on_click(on_clear)
    save_btn.on_click(on_save)

    # ── Layout ──
    toolbar = widgets.HBox(
        [send_btn, clear_btn, save_btn, status_lbl],
        layout=widgets.Layout(gap="8px", align_items="center", padding="6px 0"),
    )
    input_area = widgets.VBox([user_input, toolbar])
    ui = widgets.VBox([chat_output, input_area])

    # ── Welcome message ──
    welcome = (
        "👋 **Hello! I'm MedAssist AI (powered by Gemini).**\n\n"
        "I can help you with:\n"
        "- Understanding symptoms and medical conditions\n"
        "- Explaining medications and side effects\n"
        "- General wellness and preventive care\n"
        "- Guidance on when to seek professional help\n\n"
        "⚠️ *I'm an AI assistant, not a doctor. Always consult a "
        "licensed healthcare professional for personal medical decisions.*\n\n"
        "How can I help you today?"
    )
    conversation.add("assistant", welcome)
    chat_log.append(render_message("assistant", welcome))
    refresh_chat()

    return ui


# ─────────────────────────────────────────────────────────────
# CELL 5 — Launch the chatbot
# ─────────────────────────────────────────────────────────────
ui = build_chatbot_ui()
display(ui)
