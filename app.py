"""
app.py  — Voice-first Interview (Whisper engine)
Patched: semantic scoring + improved UI order + skip/stop truncation + safe fallbacks.
Based on the user's uploaded app.py. fileciteturn1file0
"""

import logging
import time
import html
import traceback

import gradio as gr

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

from utils.audio_utils import AudioHandler
from utils.voice_config import VoiceConfig
from models.enhanced_interview_model import EnhancedInterviewModel

# Optional semantic/fuzzy scoring libs (lazy loaded)
_semantic_model = None
_have_rapidfuzz = False
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _have_sentence_transformers = True
except Exception:
    _have_sentence_transformers = False
    try:
        from rapidfuzz import fuzz
        _have_rapidfuzz = True
    except Exception:
        _have_rapidfuzz = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_interview")

def get_semantic_model():
    """Lazy-load the sentence-transformers model if available."""
    global _semantic_model
    if not _have_sentence_transformers:
        return None
    if _semantic_model is None:
        try:
            # small model that is fast and effective
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence-transformers 'all-MiniLM-L6-v2' for semantic scoring.")
        except Exception as e:
            logger.warning("Failed to load sentence-transformers model: %s", e)
            _semantic_model = None
    return _semantic_model

def semantic_score(correct: str, user: str) -> float:
    """
    Return a 0-10 semantic similarity score using sentence-transformers if available.
    Falls back to rapidfuzz token_set_ratio scaled to 0-10, else uses strict_score logic.
    """
    if not correct or not user:
        return 0.0

    # Prefer sentence-transformers
    model = get_semantic_model()
    if model:
        try:
            emb_c = model.encode(correct, convert_to_tensor=True)
            emb_u = model.encode(user, convert_to_tensor=True)
            sim = st_util.pytorch_cos_sim(emb_c, emb_u).item()  # -1..1
            sim = max(0.0, sim)  # clamp negatives
            # Map similarity to 0..10 (tuneable)
            # sim ~ 0.0 -> 0, sim ~0.6 -> ~7, sim ~0.8 -> ~9
            score = round(min(10.0, (sim ** 1.1) * 11.0), 1)
            return score
        except Exception as e:
            logger.warning("semantic scoring failed: %s", e)

    # Fallback to rapidfuzz if available
    if _have_rapidfuzz:
        try:
            ratio = fuzz.token_set_ratio(correct, user)  # 0..100
            score = round((ratio / 100.0) * 10.0, 1)
            return score
        except Exception as e:
            logger.warning("rapidfuzz scoring failed: %s", e)

    # Final fallback: strict token overlap scoring (existing strict_score in class)
    # We'll let the class's strict_score handle it (call externally).
    return None  # caller should handle None and call strict_score

class VoiceInterviewBot:
    def __init__(self, questions_per_session: int = 5):
        cfg = VoiceConfig()  # user config: device index, whisper model size etc
        # Create audio handler (wraps TTS + STT using Whisper)
        self.audio = AudioHandler(device_index=getattr(cfg, "device_index", None),
                                  model_size=getattr(cfg, "model_size", "tiny"))
        self.model = EnhancedInterviewModel()
        self.questions_per_session = questions_per_session
        self.reset_session()

    def reset_session(self):
        self.role = None
        self.difficulty = None
        self.questions = []          # session question list (dicts)
        self.current_index = 0
        self.state = "idle"          # idle -> ask_role -> ask_difficulty -> ask_question -> summary

        # NEW FLAGS for skip / stop functionality
        self.skip_flag = False
        self.stop_flag = False

    # NEW: Skip current question (Option B)
    def skip_question(self):
        """
        Called by UI. Marks skip_flag so run_interview will mark the current question as
        skipped (user_answer="skipped", score=0) and move on.
        """
        self.skip_flag = True
        return "### ⏭ Question Skipped\n\nMoving to the next question..."

    # NEW: Stop interview now
    def stop_interview(self):
        """
        Called by UI. Marks stop_flag so run_interview will terminate ASAP and yield the summary.
        """
        self.stop_flag = True
        return "### 🛑 Interview End Requested\n\nEnding interview and generating summary..."

    # normalize many possible spoken role variations to DB role keys
    def normalize_role(self, spoken: str) -> str:
        if not spoken:
            return ""

        s = spoken.lower().strip()

        backend_words = ["backend", "back end", "server", "api", "rest api"]
        python_developer_words = ["python"," python developer"]
        data_science_words = ["data scientist", "data science"]
        data_analyst_words = ["data analyst", "analytics"]
        frontend_developer_words = ["frontend", "front end","frontend developer"]
        software_engineer_words = ["software engineer", "software developer"]
        devops_engineer_words = ["devops engineer", "dev ops", "site reliability", "sre"]

        if any(w in s for w in backend_words):
            return "backend developer"
        if any(w in s for w in python_developer_words):
            return "python developer"
        if any(w in s for w in data_science_words):
            return "data scientist"
        if any(w in s for w in data_analyst_words):
            return "data analyst"
        if any(w in s for w in frontend_developer_words):
            return "frontend developer"
        if any(w in s for w in software_engineer_words):
            return "software engineer"
        if any(w in s for w in devops_engineer_words):
            return "devops engineer"

        return "backend developer"

    def normalize_difficulty(self, text: str) -> str:
        text = text.lower().strip()

        # direct words
        if "easy" in text:
            return "easy"
        if "medium" in text or "moderate" in text or "normal" in text:
            return "medium"
        if "hard" in text or "difficult" in text or "advanced" in text or "tough" in text:
            return "hard"

        # fallback
        return "easy"

    # fallback strict scorer: exact-match substring -> 10 else 0 (user asked for "strict")
    def strict_score(self, correct: str, user: str) -> float:
        if not user or not correct:
            return 0.0
        c = correct.lower().strip()
        u = user.lower().strip()
        # if user includes some significant token from correct answer -> partial score
        if u == c:
            return 10.0
        # check overlap of important words
        c_words = {w for w in c.split() if len(w) > 3}
        u_words = {w for w in u.split() if len(w) > 3}
        if not c_words:
            return 0.0
        overlap = c_words.intersection(u_words)
        if len(overlap) == 0:
            return 0.0
        # score proportionally but scaled strictly
        score = round(min(10.0, 10.0 * (len(overlap) / len(c_words))), 1)
        # if overlap very small, treat as zero to be strict
        return score if score >= 6.0 else 0.0

    # generate summary Markdown (only for this session)
    def generate_summary_md(self) -> str:
        total = len(self.questions)
        if total == 0:
            avg = 0.0
        else:
            avg = sum(q.get("score", 0.0) for q in self.questions) / total
        md = f"## Interview Summary\n\n- Role: **{html.escape(str(self.role))}**\n- Difficulty: **{html.escape(str(self.difficulty))}**\n- Questions: **{total}**\n- Average Score: **{avg:.1f}/10**\n\n---\n\n"
        for i, q in enumerate(self.questions, 1):
            md += f"### Q{i}: {html.escape(q.get('question',''))}\n\n"
            md += f"**Your Answer:**  \n{html.escape(q.get('user_answer','unknown'))}\n\n"
            md += f"**Correct Answer:**  \n{html.escape(q.get('correct_answer',''))}\n\n"
            md += f"**Score:** {q.get('score',0.0)}/10\n\n---\n\n"
        return md

    # The main generator: used by Gradio button .click(fn=bot.run_interview, ...)
    def run_interview(self):
        """
        Generator that progresses the interview step-by-step.
        Each yield updates the markdown output.
        """
        try:
            # reset between sessions
            self.reset_session()
            self.state = "ask_role"

            # 1) Ask role
            # UI first, then speak (so UI updates instantly)
            yield " Listening for role..."
            self.audio.speak("Which role would you like? For example: Python developer or Data Scientist.")
            role_spoken = self.audio.listen(timeout=12)
            logger.info("Raw role (STT): %r", role_spoken)

            if not role_spoken:
                self.audio.speak("I couldn't hear role. Defaulting to backend developer.")
                self.role = "backend developer"
            else:
                self.role = self.normalize_role(role_spoken)
            logger.info("Normalized role -> %s", self.role)
            yield f"Role selected: **{self.role}**"
            self.audio.speak(f"You selected {self.role}. Now say easy, medium or hard for difficulty.")

            # 2) Ask difficulty
            yield " Listening for difficulty…"
            diff_spoken = self.audio.listen(timeout=8)
            logger.info("Raw difficulty (STT): %r", diff_spoken)
            if not diff_spoken:
                self.audio.speak("I couldn't hear difficulty. Defaulting to easy.")
                self.difficulty = "easy"
            else:
                self.difficulty = self.normalize_difficulty(diff_spoken)
            logger.info("Normalized difficulty -> %s", self.difficulty)
            yield f"Difficulty: **{self.difficulty}** — Starting interview…"
            self.audio.speak(f"Difficulty set to {self.difficulty}. I will now ask {self.questions_per_session} questions.")

            # 3) Load questions from model DB — only current session
            try:
                self.questions = self.model.get_questions(self.role, self.difficulty, self.questions_per_session) or []
            except Exception as e:
                logger.warning("EnhancedInterviewModel.get_questions failed: %s", e)
                # fallback: try a generic call
                try:
                    self.questions = self.model.get_questions(self.role, self.difficulty) or []
                except Exception:
                    logger.error("Cannot fetch questions from model; aborting. %s", traceback.format_exc())
                    self.questions = []

            if not self.questions:
                self.audio.speak("I couldn't find questions for that role and difficulty. Please try another role or difficulty.")
                yield " No questions found for your selection. Please restart and try a different role or difficulty."
                return

            # ensure each question has expected fields
            for q in self.questions:
                q.setdefault("question", "")
                q.setdefault("answer", "")           # canonical/correct answer in DB (if present)
                q.setdefault("user_answer", "")
                q.setdefault("score", 0.0)
                q.setdefault("correct_answer", q.get("answer", ""))

            # 4) Ask questions one by one
            for idx, q in enumerate(self.questions):
                # END CHECK — if stop_flag set before processing this question, yield summary immediately
                if self.stop_flag:
                    # Keep only questions up to this point (previously asked)
                    # Since we're at start of this loop before asking idx, only previous questions were asked:
                    self.questions = self.questions[:idx]
                    summary_md = self.generate_summary_md()
                    self.audio.speak("Interview ended. Generating summary.")
                    yield summary_md
                    return

                # SKIP CHECK — if skip_flag set before asking this question
                if self.skip_flag:
                    self.skip_flag = False
                    q["user_answer"] = "skipped"
                    q["score"] = 0.0
                    q["correct_answer"] = q.get("correct_answer", q.get("answer", ""))
                    self.audio.speak(f"Question {idx+1} skipped.")
                    yield f"Q{idx+1}: {q['question']}\n\nStatus: Skipped. Score: 0/10"
                    continue

                self.current_index = idx
                qtext = q["question"]
                # ask — UI updates first, then TTS
                yield f"Q{idx+1}: {qtext}\n\n Listening for your answer…"
                self.audio.speak(f"Question {idx+1}. {qtext}")

                # listen
                answer = self.audio.listen(timeout=20)

                # If stop requested while listening (user clicked End Interview), handle immediately
                if self.stop_flag:
                    # Keep questions up to and including the current one
                    #self.questions = self.questions[:idx+1]
                    summary_md = self.generate_summary_md()
                    self.audio.speak("Interview ended. Generating summary.")
                    yield summary_md
                    return

                # If skip requested while listening, mark this question as skipped
                if self.skip_flag:
                    self.skip_flag = False
                    q["user_answer"] = "skipped"
                    q["score"] = 0.0
                    q["correct_answer"] = q.get("correct_answer", q.get("answer", ""))
                    self.audio.speak("Question skipped.")
                    yield f"Q{idx+1}: {qtext}\n\nStatus: Skipped. Score: 0/10"
                    continue

                logger.info("Answer (raw) Q%d: %r", idx+1, answer)
                # store
                q["user_answer"] = answer if answer else "unknown"
                # attempt to score using model's evaluation if available
                scored = False
                try:
                    # model.evaluate_answer should ideally update q with 'score' and 'correct_answer'
                    if hasattr(self.model, "evaluate_answer"):
                        self.model.evaluate_answer(q, answer)
                        # ensure we have numeric score
                        if isinstance(q.get("score", None), (int, float)):
                            scored = True
                except Exception as e:
                    logger.warning("model.evaluate_answer threw: %s", e)

                # fallback semantic / fuzzy / strict scoring
                if not scored:
                    q["correct_answer"] = q.get("correct_answer", "")
                    # Try semantic_score
                    sscore = semantic_score(q["correct_answer"], q["user_answer"])
                    if sscore is None:
                        # semantic_score returned None meaning no semantic/fuzzy libraries available
                        q["score"] = self.strict_score(q["correct_answer"], q["user_answer"])
                    else:
                        q["score"] = sscore

                # provide brief feedback (UI first, then speak)
                feedback = f"Saved answer. Score: {q['score']}/10"
                logger.info("Q%d stored - score %s", idx+1, q["score"])
                yield f"Q{idx+1}: {qtext}\n\nYour Answer: {q['user_answer']}\n\nScore: {q['score']}/10"
                self.audio.speak(feedback)

            # 5) Summary
            summary_md = self.generate_summary_md()
            # show summary immediately, then speak average
            yield summary_md
            self.audio.speak("The interview is complete. I will read out your average score.")
            avg_score = 0.0
            if len(self.questions):
                avg_score = sum(q.get("score",0) for q in self.questions)/len(self.questions)
            self.audio.speak(f"Your average score is {avg_score:.1f} out of 10.")
            return

        except Exception as ex:
            logger.exception("Error during interview run: %s", ex)
            self.audio.speak("An error occurred during the interview. Please check the application logs.")
            yield " An unexpected error occurred. See console for details."

# ----------------------------
# Enhanced Gradio UI with Modern Design
# ----------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

/* Main background with animated gradient */
body {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f3460 100%);
    background-size: 200% 200%;
    animation: gradientShift 15s ease infinite;
    color: #e6e6e6;
    font-family: 'Inter', sans-serif;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Container */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Header with glow effect */
#title {
    font-size: 42px !important;
    text-align: center;
    padding: 30px 20px;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    text-shadow: 0px 0px 30px rgba(102, 126, 234, 0.4);
    animation: titlePulse 3s ease-in-out infinite;
}

@keyframes titlePulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
}

#subtitle {
    text-align: center;
    color: #a0a0b0;
    font-size: 16px;
    margin-top: -15px;
    margin-bottom: 30px;
}

/* Glass morphism card */
.card-box {
    background: rgba(255, 255, 255, 0.08);
    padding: 40px;
    border-radius: 24px;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease;
}

.card-box:hover {
    transform: translateY(-2px);
}

/* Button container */
.button-row {
    display: flex;
    gap: 20px;
    margin-bottom: 30px;
}

/* Modern buttons with icons and shadows */
.gr-button {
    border-radius: 16px !important;
    font-size: 18px !important;
    padding: 18px 32px !important;
    font-weight: 700 !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: relative;
    overflow: hidden;
}

.gr-button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.gr-button:active::before {
    width: 300px;
    height: 300px;
}

#start-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
}

#start-btn:hover {
    background: linear-gradient(135deg, #5568d3 0%, #65408b 100%) !important;
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
}

#skip-btn {
    background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%) !important;
    color: white !important;
    box-shadow: 0 10px 30px rgba(255, 167, 38, 0.4);
}

#skip-btn:hover {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%) !important;
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(255, 167, 38, 0.6);
}

#stop-btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important;
    box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
}

#stop-btn:hover {
    background: linear-gradient(135deg, #e082ea 0%, #e4465b 100%) !important;
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(245, 87, 108, 0.6);
}

/* Output area with modern styling */
.output-md {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    padding: 28px;
    border-radius: 18px;
    margin-top: 25px;
    border: 1px solid rgba(102, 126, 234, 0.2);
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.2);
    min-height: 120px;
    transition: all 0.3s ease;
}

.output-md:hover {
    border-color: rgba(102, 126, 234, 0.4);
    box-shadow: 
        inset 0 2px 8px rgba(0, 0, 0, 0.2),
        0 0 20px rgba(102, 126, 234, 0.2);
}

.output-md h3 {
    color: #667eea;
    font-weight: 700;
    margin-bottom: 12px;
}

.output-md p {
    line-height: 1.7;
    color: #d0d0d8;
}

/* Status indicator */
.status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.1); }
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5568d3 0%, #65408b 100%);
}
"""

def build_ui():
    bot = VoiceInterviewBot()

    with gr.Blocks(css=custom_css, title="AI Voice Interview Bot", theme=gr.themes.Soft()) as demo:

        gr.Markdown("<h1 id='title'> AI Voice Interview Bot</h1>")
        #gr.Markdown("<p id='subtitle'>Experience seamless voice-powered interviews with AI</p>")

        with gr.Column(elem_classes="card-box"):

            with gr.Row(elem_classes="button-row"):
                start_btn = gr.Button(" Start Interview", elem_id="start-btn", scale=1)
                skip_btn = gr.Button(" Skip Question", elem_id="skip-btn", scale=1)
                stop_btn = gr.Button(" End Interview", elem_id="stop-btn", scale=1)

            output_md = gr.Markdown(
                "###  Ready to Begin\n\nClick **Start Interview** to begin your AI-powered voice interview session.",
                elem_classes="output-md"
            )

        # Generator start
        start_btn.click(
            fn=bot.run_interview,
            inputs=None,
            outputs=output_md
        )

        # Skip current question
        skip_btn.click(
            fn=bot.skip_question,
            inputs=None,
            outputs=output_md
        )


        # End interview permanently
        stop_btn.click(
            fn=bot.stop_interview,
            inputs=None,
            outputs=output_md
        )


    return demo


if __name__ == "__main__":
    import os

    ui = build_ui()

    try:
        # Render provides PORT automatically (example: 5023)
        port = int(os.environ.get("PORT", 7860))

        ui.queue()  # needed for streaming

        ui.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False  # MUST be False on Render
        )

    except Exception as e:
        print("Failed to launch Gradio app:", e)


