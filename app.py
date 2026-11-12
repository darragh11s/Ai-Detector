import difflib
import html
import io
import re
import statistics
import docx
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from huggingface_hub import hf_hub_download
import pypdf
from safetensors.torch import load_file as load_safetensors
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

# --- Page Configuration ---
st.set_page_config(page_title="Document Diff & AI Detector", layout="wide")


# --- Custom DeBERTa Model Architecture (Bypasses Transformers Meta-Init Bug) ---
class DesklibAIDetectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        return logits


@st.cache_resource
def load_detector(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    model = DesklibAIDetectionModel(config)

    # Download weights directly via huggingface_hub and load standard state_dict
    try:
        weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state_dict = load_safetensors(weights_path, device="cpu")
    except Exception:
        weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Clean potential prefix differences if needed
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


# --- Word (.docx) Block Traversal (Paragraphs + Grid & Merged Tables in Order) ---
def iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Unsupported parent element type")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def extract_table_blocks(table: Table) -> list[str]:
    blocks = []
    seen_tcs = set()
    for row in table.rows:
        row_cells = []
        for cell in row.cells:
            # Avoid duplicating merged cells
            if cell._tc in seen_tcs:
                continue
            seen_tcs.add(cell._tc)
            cell_parts = []
            for block in iter_block_items(cell):
                if isinstance(block, Paragraph):
                    text = block.text.strip()
                    if text:
                        cell_parts.append(text)
                elif isinstance(block, Table):
                    cell_parts.extend(extract_table_blocks(block))
            if cell_parts:
                row_cells.append(" ".join(cell_parts))
        if row_cells:
            blocks.append(" | ".join(row_cells))
    return blocks


def extract_docx_blocks(raw_bytes: bytes) -> list[str]:
    doc = docx.Document(io.BytesIO(raw_bytes))
    blocks = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                blocks.append(text)
        elif isinstance(block, Table):
            blocks.extend(extract_table_blocks(block))
    return blocks


def extract_blocks(uploaded_file) -> list[str]:
    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if filename.endswith(".docx"):
        return extract_docx_blocks(raw_bytes)
    elif filename.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="ignore")
        return [line.strip() for line in text.splitlines() if line.strip()]
    elif filename.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        lines = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            for line in page_text.splitlines():
                if line.strip():
                    lines.append(line.strip())
        return lines
    return []


# --- Diff & Isolation Logic ---
def clean_template_overlap(orig_chunk: str, filled_chunk: str) -> str:
    """Strips unchanged template labels/prompts from a modified cell or paragraph."""
    orig_words = orig_chunk.split()
    filled_words = filled_chunk.split()
    matcher = difflib.SequenceMatcher(None, orig_words, filled_words)
    added_words = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added_words.extend(filled_words[j1:j2])
    return " ".join(added_words).strip()


def get_added_segments(orig_blocks: list[str], filled_blocks: list[str], min_words: int) -> list[str]:
    matcher = difflib.SequenceMatcher(None, orig_blocks, filled_blocks)
    added_segments = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            for block in filled_blocks[j1:j2]:
                if len(block.split()) >= min_words:
                    added_segments.append(block)
        elif tag == "replace":
            orig_span = " ".join(orig_blocks[i1:i2])
            filled_span = " ".join(filled_blocks[j1:j2])
            isolated = clean_template_overlap(orig_span, filled_span)
            if len(isolated.split()) >= min_words:
                added_segments.append(isolated)

    return added_segments


# --- Burstiness & Chunked AI Probability Scoring ---
def compute_burstiness(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip().split()) > 1]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(lengths)
    if mean_len == 0:
        return 0.0
    return round(statistics.stdev(lengths) / mean_len, 2)


def score_text(text: str, tokenizer, model, device) -> float:
    words = text.split()
    if not words:
        return 0.0

    # Overlapping 380-word chunks so long answers are scored completely without truncation
    chunk_size = 380
    overlap = 60
    chunks = []
    if len(words) <= chunk_size:
        chunks.append(" ".join(words))
    else:
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + chunk_size])
            if chunk:
                chunks.append(chunk)

    probs = []
    with torch.no_grad():
        for chunk in chunks:
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            prob = torch.sigmoid(logits).squeeze().item()
            probs.append(prob * 100.0)

    return round(sum(probs) / len(probs), 1)


def get_tier_style(ai_prob: float):
    if ai_prob >= 80.0:
        return {
            "label": "Definitely AI",
            "emoji": "🔴",
            "bg": "#ffe6e6",
            "border": "#d93025",
            "text": "#7a0c0c",
            "flagged": True,
        }
    elif ai_prob >= 55.0:
        return {
            "label": "Likely AI",
            "emoji": "🟠",
            "bg": "#fff0e0",
            "border": "#e67c00",
            "text": "#7a3e00",
            "flagged": True,
        }
    elif ai_prob >= 30.0:
        return {
            "label": "Possibly AI",
            "emoji": "🟡",
            "bg": "#fffbe6",
            "border": "#f4b400",
            "text": "#664d00",
            "flagged": True,
        }
    else:
        return {
            "label": "Human",
            "emoji": "🟢",
            "bg": "#e6f4ea",
            "border": "#1e8e3e",
            "text": "#0d4a22",
            "flagged": False,
        }


def render_highlighted_card(index: int, text: str, ai_prob: float, burstiness: float):
    tier = get_tier_style(ai_prob)
    safe_text = html.escape(text).replace("\n", "<br>")
    card_html = f"""
    <div style="
        background-color: {tier['bg']};
        border-left: 6px solid {tier['border']};
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
        color: #1f1f1f;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-weight: 700;
            color: {tier['text']};
        ">
            <span>{tier['emoji']} Section {index} — {tier['label']} ({ai_prob}%)</span>
            <span style="font-size: 0.85em; font-weight: 500;">
                Words: {len(text.split())} | Sentence Burstiness: {burstiness}
            </span>
        </div>
        <div style="font-size: 0.98em; line-height: 1.55;">
            {safe_text}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# --- Sidebar Controls ---
with st.sidebar:
    st.header("Detection Settings")
    model_choice = st.selectbox(
        "Open-Source Detection Model",
        options=[
            "desklib/ai-text-detector-v1.01",
            "desklib/ai-text-detector-academic-v1.01",
        ],
        index=0,
    )
    min_words = st.slider(
        "Minimum Words per Added Section",
        min_value=3,
        max_value=30,
        value=6,
        help="Filters out short form fields like names, dates, and checkboxes.",
    )
    st.divider()
    st.markdown(
        """
        **Highlight Legend:**
        * 🟢 **Human:** `< 30%`
        * 🟡 **Possibly AI:** `30% – 54.9%`
        * 🟠 **Likely AI:** `55% – 79.9%`
        * 🔴 **Definitely AI:** `80% – 100%`
        """
    )

# --- Main Interface ---
st.title("Document Diff & AI Content Detector")
st.write(
    "Upload the **original blank template** and the **completed document** (Word `.docx` tables & paragraphs supported) to isolate and highlight all added AI content."
)

try:
    with st.spinner(f"Loading model ({model_choice})..."):
        tokenizer, model, device = load_detector(model_choice)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    orig_file = st.file_uploader(
        "1. Original Template Document", type=["docx", "pdf", "txt"]
    )
with col2:
    filled_file = st.file_uploader(
        "2. Completed / Filled-Out Document", type=["docx", "pdf", "txt"]
    )

if orig_file and filled_file:
    if st.button("Scan Added Content", type="primary"):
        with st.spinner("Extracting Word tables/paragraphs and analyzing added text..."):
            orig_blocks = extract_blocks(orig_file)
            filled_blocks = extract_blocks(filled_file)
            added_segments = get_added_segments(orig_blocks, filled_blocks, min_words)

            if not added_segments:
                st.warning(
                    f"No new sections with at least {min_words} words were detected. Try lowering the minimum word threshold in the sidebar."
                )
            else:
                # 1. Combined Full-Text Analysis
                combined_text = "\n\n".join(added_segments)
                combined_prob = score_text(combined_text, tokenizer, model, device)
                combined_tier = get_tier_style(combined_prob)
                combined_burstiness = compute_burstiness(combined_text)

                # 2. Section-by-Section Analysis
                results = []
                for idx, seg in enumerate(added_segments, 1):
                    prob = score_text(seg, tokenizer, model, device)
                    burst = compute_burstiness(seg)
                    tier = get_tier_style(prob)
                    results.append(
                        {
                            "index": idx,
                            "text": seg,
                            "prob": prob,
                            "burst": burst,
                            "tier": tier,
                        }
                    )

                flagged_results = [r for r in results if r["tier"]["flagged"]]
                avg_section_prob = round(
                    sum(r["prob"] for r in results) / len(results), 1
                )

                # --- Summary Metrics ---
                st.subheader("Detection Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Combined Added Text Score",
                    f"{combined_prob}%",
                    f"{combined_tier['emoji']} {combined_tier['label']}",
                )
                m2.metric("Average Section Score", f"{avg_section_prob}%")
                m3.metric(
                    "Flagged Sections (Possibly+ AI)",
                    f"{len(flagged_results)} / {len(results)}",
                )
                m4.metric("Combined Burstiness", f"{combined_burstiness}")

                st.divider()

                # --- Highlighted Results Tabs ---
                tab_all, tab_flagged = st.tabs(
                    [
                        f"All Added Sections ({len(results)})",
                        f"Flagged Only — Possibly / Likely / Definitely AI ({len(flagged_results)})",
                    ]
                )

                with tab_all:
                    for r in results:
                        render_highlighted_card(
                            r["index"], r["text"], r["prob"], r["burst"]
                        )

                with tab_flagged:
                    if not flagged_results:
                        st.success("All added sections scored as Human (< 30%).")
                    else:
                        for r in flagged_results:
                            render_highlighted_card(
                                r["index"], r["text"], r["prob"], r["burst"]
                            )
