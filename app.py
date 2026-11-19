
import time
import statistics
from collections import deque
from html import escape

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prefix_kv_cache import PrefixKVCache, CachedGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Prefix KV Cache | Neural Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CACHE CONFIGURATION
# ============================================================
CACHE_CAPACITY = 50
CACHE_MEMORY_MB = 512


# ============================================================
# FUTURISTIC DARK UI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 40%, #111827 100%);
    color: #e0f2fe;
    font-family: 'Rajdhani', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1px;
}

#MainMenu, footer, header {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
    border-right: 1px solid #1e3a5f;
}

.metric-card {
    background: linear-gradient(145deg, rgba(15,23,42,.9), rgba(30,41,59,.7));
    border: 1px solid rgba(0,240,255,.25);
    border-radius: 16px;
    padding: 18px 12px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,240,255,.08);
    min-height: 105px;
}

.metric-label {
    font-size: .78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    font-family: 'Orbitron', sans-serif;
}

.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00f0ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Orbitron', sans-serif;
    overflow-wrap: anywhere;
}

.metric-sub {
    font-size: .75rem;
    color: #64748b;
    margin-top: 4px;
}

.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.2rem;
    background: linear-gradient(90deg, #00f0ff, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .8rem;
    letter-spacing: 2px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15,23,42,.6);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid rgba(0,240,255,.15);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #94a3b8;
    font-family: 'Orbitron', sans-serif;
    font-size: .78rem;
    padding: 10px 14px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,240,255,.15), rgba(191,0,255,.1)) !important;
    color: #00f0ff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0e7490, #7c3aed) !important;
    color: white !important;
    border: 1px solid rgba(0,240,255,.4) !important;
    border-radius: 10px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: .5px !important;
    box-shadow: 0 0 15px rgba(0,240,255,.2) !important;
}

.stTextArea textarea, .stTextInput input {
    background: rgba(15,23,42,.8) !important;
    border: 1px solid rgba(0,240,255,.2) !important;
    border-radius: 12px !important;
    color: #e0f2fe !important;
}

.glass-panel {
    background: rgba(15,23,42,.6);
    border: 1px solid rgba(0,240,255,.2);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,.3);
}

.main-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.35rem;
    background: linear-gradient(90deg, #00f0ff, #a855f7, #00f0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: 3px;
    margin-bottom: .2rem;
}

.main-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    letter-spacing: 2px;
    margin-bottom: 2rem;
}

.output-heading {
    color: #00f0ff;
    font-family: 'Orbitron', sans-serif;
    font-size: .9rem;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.output-box {
    background: rgba(15,23,42,.85);
    border: 1px solid rgba(0,240,255,.25);
    border-radius: 12px;
    padding: 14px;
    min-height: 130px;
    max-height: 280px;
    overflow-y: auto;
    color: #e0f2fe;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: .75rem;
    font-family: 'Orbitron', sans-serif;
    color: #00f0ff;
    background: rgba(0,240,255,.15);
    border: 1px solid rgba(0,240,255,.4);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL AND CACHE LOADER
# ============================================================
@st.cache_resource(show_spinner="⚡ Initializing neural core...")
def load_generator(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Actual active cache: 50 entries, 512 MB
    cache = PrefixKVCache(
        max_entries=CACHE_CAPACITY,
        max_memory_mb=CACHE_MEMORY_MB
    )

    generator = CachedGenerator(
        model=model,
        tokenizer=tokenizer,
        cache=cache
    )

    return generator, device


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_cache_stats(cache):

    stats = getattr(cache, "stats", {})

    if callable(stats):
        stats = stats()

    if isinstance(stats, dict):
        return stats

    if hasattr(stats, "__dict__"):
        return vars(stats)

    return {}


def metric(stats, name, default=0):
    value = stats.get(name, default)
    return default if value is None else value


def format_memory(value):
    try:
        return f"{float(value) / (1024 ** 2):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def run_generation(
    generator,
    system_prompt,
    user_message,
    max_tokens,
    use_cache
):

    synchronize_gpu()
    start = time.perf_counter()

    output = generator.generate(
        system_prompt=system_prompt,
        user_message=user_message,
        max_new_tokens=max_tokens,
        use_cache=use_cache
    )

    synchronize_gpu()

    elapsed = (time.perf_counter() - start) * 1000

    return output, elapsed


def render_metric_card(label, value, sub=""):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def safe_stats(values):

    if not values:
        return 0.0, 0.0, 0.0

    ordered = sorted(values)

    p95_index = min(
        len(ordered) - 1,
        max(0, int(0.95 * len(ordered)))
    )

    return (
        statistics.mean(values),
        statistics.median(values),
        ordered[p95_index]
    )


def render_output(label, output, color="#00f0ff"):

    safe_output = escape(str(output))

    st.markdown(f"""
    <div class="output-heading" style="color:{color};">
        {label}
    </div>
    <div class="output-box">{safe_output}</div>
    """, unsafe_allow_html=True)


def reset_arena():

    st.session_state.generation_history = []

    st.session_state.latency_log = {
        "cached": deque(maxlen=50),
        "uncached": deque(maxlen=50)
    }


# ============================================================
# SESSION STATE
# ============================================================
if "pinned_prompts" not in st.session_state:
    st.session_state.pinned_prompts = []

if "generation_history" not in st.session_state:
    st.session_state.generation_history = []

if "latency_log" not in st.session_state:
    st.session_state.latency_log = {
        "cached": deque(maxlen=50),
        "uncached": deque(maxlen=50)
    }

if "latest_inference" not in st.session_state:
    st.session_state.latest_inference = None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown("### ⚙️ CONTROL PANEL")
    st.markdown("---")

    model_name = st.text_input(
        "Model",
        value="gpt2",
        help="Hugging Face model ID, for example gpt2 or distilgpt2."
    )

    max_tokens = st.slider(
        "Max new tokens",
        8,
        128,
        32,
        8
    )

    st.markdown(f"**Cache capacity:** `{CACHE_CAPACITY}` entries")
    st.markdown(f"**Cache memory limit:** `{CACHE_MEMORY_MB} MB`")

    if st.button("🔄 Reload Model", use_container_width=True):

        load_generator.clear()

        st.session_state.latest_inference = None
        st.session_state.pinned_prompts = []

        st.rerun()

    st.markdown("---")
    st.markdown("### 📡 SYSTEM STATUS")


# ============================================================
# LOAD MODEL
# ============================================================
try:

    generator, device = load_generator(model_name)

except Exception as error:

    st.error(f"Neural core failed to load: {error}")
    st.stop()


with st.sidebar:

    st.markdown(
        '<div style="margin-top:10px;">'
        '<span class="status-pill">● ONLINE</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"**Model:** `{model_name}`")
    st.markdown(f"**Device:** `{device.upper()}`")

    if torch.cuda.is_available():

        st.markdown(
            f"**GPU:** `{torch.cuda.get_device_name(0)}`"
        )


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">⚡ PREFIX KV CACHE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'NEURAL INFERENCE OPTIMIZATION CORE'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LIVE CACHE METRICS
# ============================================================
stats = get_cache_stats(generator.cache)

entries = metric(stats, "entries")
capacity = metric(stats, "capacity")
hits = metric(stats, "hits")
misses = metric(stats, "misses")
evictions = metric(stats, "evictions")

pinned = metric(stats, "pinned_entries")
ordinary = metric(stats, "ordinary_entries")

hit_rate = metric(stats, "hit_rate")
memory_bytes = metric(stats, "memory_bytes")

if isinstance(hit_rate, (int, float)):

    if hit_rate <= 1:
        hit_rate_display = f"{hit_rate * 100:.1f}%"
    else:
        hit_rate_display = f"{hit_rate:.1f}%"

else:
    hit_rate_display = "0.0%"


st.markdown(
    '<div class="section-header">◉ LIVE CACHE METRICS</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    render_metric_card("ENTRIES", f"{entries}/{capacity}", "capacity")

with c2:
    render_metric_card("HITS", hits, "cache hits")

with c3:
    render_metric_card("HIT RATE", hit_rate_display, "efficiency")

with c4:
    render_metric_card("PINNED", pinned, "protected")

with c5:
    render_metric_card(
        "MEMORY",
        f"{format_memory(memory_bytes)} MB",
        "KV tensors"
    )

with c6:
    render_metric_card("EVICTIONS", evictions, "LRU removed")


st.caption(
    f"Ordinary entries: {ordinary}  |  "
    f"Misses: {misses}  |  "
    f"Cache requests: {metric(stats, 'requests')}"
)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_lab, tab_vault, tab_arena, tab_gpu = st.tabs([
    "🧪 INFERENCE LAB",
    "🔐 PREFIX VAULT",
    "📊 BENCHMARK ARENA",
    "🖥️ GPU MONITOR"
])


# ============================================================
# TAB 1: INFERENCE LAB
# ============================================================
with tab_lab:

    st.markdown(
        '<div class="section-header">'
        'INTERACTIVE INFERENCE LAB'
        '</div>',
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns(
        [1, 1.15],
        gap="large"
    )

    with col_left:

        st.markdown("#### 📥 INPUT PROMPTS")

        system_prompt = st.text_area(
            "System Prompt (Reusable Prefix)",
            value=(
                "You are a helpful AI assistant. "
                "Answer clearly and concisely."
            ),
            height=130,
            key="lab_system_prompt"
        )

        user_message = st.text_area(
            "User Message",
            value="Explain what a KV cache is in a Transformer model.",
            height=110,
            key="lab_user_message"
        )

        generation_mode = st.radio(
            "Generation Mode",
            ["Cached", "Uncached", "Compare both"],
            horizontal=True,
            key="lab_generation_mode"
        )

        run_btn = st.button(
            "⚡ RUN INFERENCE",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------
    if run_btn:

        if not system_prompt.strip() or not user_message.strip():

            st.warning("Enter both the system prompt and user message.")

        else:

            try:

                result = {
                    "mode": generation_mode,
                    "system_prompt": system_prompt,
                    "user_message": user_message
                }

                with st.spinner("Neural cores processing..."):

                    if generation_mode == "Cached":

                        output, elapsed = run_generation(
                            generator,
                            system_prompt,
                            user_message,
                            max_tokens,
                            True
                        )

                        result["cached_output"] = output
                        result["cached_time"] = elapsed

                        st.session_state.generation_history.append({
                            "Mode": "Cached",
                            "Latency (ms)": round(elapsed, 2)
                        })

                        st.session_state.latency_log["cached"].append(
                            elapsed
                        )

                    elif generation_mode == "Uncached":

                        output, elapsed = run_generation(
                            generator,
                            system_prompt,
                            user_message,
                            max_tokens,
                            False
                        )

                        result["uncached_output"] = output
                        result["uncached_time"] = elapsed

                        st.session_state.generation_history.append({
                            "Mode": "Uncached",
                            "Latency (ms)": round(elapsed, 2)
                        })

                        st.session_state.latency_log["uncached"].append(
                            elapsed
                        )

                    else:

                        # Warm-up
                        run_generation(
                            generator,
                            system_prompt,
                            user_message,
                            max_tokens,
                            False
                        )

                        # Uncached measurement
                        uncached_output, uncached_time = run_generation(
                            generator,
                            system_prompt,
                            user_message,
                            max_tokens,
                            False
                        )

                        # Cached measurement
                        cached_output, cached_time = run_generation(
                            generator,
                            system_prompt,
                            user_message,
                            max_tokens,
                            True
                        )

                        result.update({
                            "uncached_output": uncached_output,
                            "cached_output": cached_output,
                            "uncached_time": uncached_time,
                            "cached_time": cached_time
                        })

                        st.session_state.generation_history.extend([
                            {
                                "Mode": "Uncached",
                                "Latency (ms)": round(uncached_time, 2)
                            },
                            {
                                "Mode": "Cached",
                                "Latency (ms)": round(cached_time, 2)
                            }
                        ])

                        st.session_state.latency_log["uncached"].append(
                            uncached_time
                        )

                        st.session_state.latency_log["cached"].append(
                            cached_time
                        )

                st.session_state.latest_inference = result

                # Refresh metrics after generation
                st.rerun()

            except Exception as error:

                st.error(f"Inference failed: {error}")

    # --------------------------------------------------------
    # OUTPUT PANEL
    # --------------------------------------------------------
    with col_right:

        st.markdown("#### 🤖 GENERATED AI RESPONSE")

        result = st.session_state.latest_inference

        if result is None:

            st.info(
                "Your generated response will appear here after "
                "you run inference."
            )

        else:

            st.caption(
                f"Mode: {result['mode']}  |  "
                f"Model: {model_name}"
            )

            if result["mode"] == "Cached":

                render_output(
                    "⚡ CACHED AI RESPONSE",
                    result["cached_output"]
                )

                st.metric(
                    "Cached latency",
                    f"{result['cached_time']:.2f} ms"
                )

            elif result["mode"] == "Uncached":

                render_output(
                    "◉ UNCACHED AI RESPONSE",
                    result["uncached_output"],
                    "#f87171"
                )

                st.metric(
                    "Uncached latency",
                    f"{result['uncached_time']:.2f} ms"
                )

            else:

                render_output(
                    "◉ UNCACHED AI RESPONSE",
                    result["uncached_output"],
                    "#f87171"
                )

                st.metric(
                    "Uncached latency",
                    f"{result['uncached_time']:.2f} ms"
                )

                st.markdown("---")

                render_output(
                    "⚡ CACHED AI RESPONSE",
                    result["cached_output"]
                )

                st.metric(
                    "Cached latency",
                    f"{result['cached_time']:.2f} ms"
                )

    # --------------------------------------------------------
    # PERFORMANCE COMPARISON
    # --------------------------------------------------------
    result = st.session_state.latest_inference

    if result is not None:

        st.divider()

        st.markdown("### 📊 PERFORMANCE COMPARISON")

        if result["mode"] == "Compare both":

            uncached_time = result["uncached_time"]
            cached_time = result["cached_time"]

            speedup = (
                uncached_time / cached_time
                if cached_time > 0 else 0
            )

            reduction = (
                (uncached_time - cached_time) / uncached_time * 100
                if uncached_time > 0 else 0
            )

            outputs_match = (
                result["uncached_output"] == result["cached_output"]
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                render_metric_card("SPEEDUP", f"{speedup:.2f}×")

            with m2:
                render_metric_card(
                    "LATENCY REDUCTION",
                    f"{reduction:.1f}%"
                )

            with m3:
                render_metric_card(
                    "OUTPUT",
                    "MATCH" if outputs_match else "DIFFERENT"
                )

            if outputs_match:
                st.success("Cached and uncached outputs match.")
            else:
                st.warning("The generated outputs differ.")

        elif result["mode"] == "Cached":

            st.info(
                f"Cached inference completed in "
                f"{result['cached_time']:.2f} ms."
            )

        else:

            st.info(
                f"Uncached inference completed in "
                f"{result['uncached_time']:.2f} ms."
            )

        with st.expander("📋 View input prompts used for this inference"):

            st.markdown("**System Prompt**")
            st.code(result["system_prompt"])

            st.markdown("**User Message**")
            st.code(result["user_message"])


# ============================================================
# TAB 2: PREFIX VAULT
# ============================================================
with tab_vault:

    st.markdown(
        '<div class="section-header">'
        'PREFIX VAULT — PINNED KV STATES'
        '</div>',
        unsafe_allow_html=True
    )

    col_pin, col_list = st.columns([1, 1.2])

    with col_pin:

        st.markdown("#### Pin New Prefix")

        prompt_to_pin = st.text_area(
            "System prompt to pin",
            value=(
                "You are a helpful AI assistant. "
                "Answer clearly and concisely."
            ),
            height=140,
            key="prompt_to_pin"
        )

        if st.button(
            "🔒 PIN TO VAULT",
            type="primary",
            use_container_width=True
        ):

            if not prompt_to_pin.strip():

                st.warning("Enter a system prompt first.")

            elif prompt_to_pin in st.session_state.pinned_prompts:

                st.warning("This prompt is already in the vault.")

            else:

                try:

                    with st.spinner(
                        "Computing KV states and pinning..."
                    ):

                        generator.pin_system_prompt(prompt_to_pin)

                    st.session_state.pinned_prompts.append(
                        prompt_to_pin
                    )

                    st.success("Prefix locked in vault.")
                    st.rerun()

                except Exception as error:

                    st.error(f"Pin failed: {error}")

    with col_list:

        st.markdown("#### Active Vault")

        if not st.session_state.pinned_prompts:

            st.markdown("""
            <div class="glass-panel" style="text-align:center;color:#64748b;">
                Vault empty — pin a system prompt to protect it from LRU eviction.
            </div>
            """, unsafe_allow_html=True)

        else:

            for index, prompt in enumerate(
                st.session_state.pinned_prompts
            ):

                st.markdown(f"""
                <div class="glass-panel" style="margin-bottom:10px;">
                    <div style="color:#a855f7;font-family:Orbitron;
                                font-size:.85rem;margin-bottom:6px;">
                        PINNED #{index + 1}
                    </div>
                    <div style="color:#cbd5e1;font-size:.95rem;">
                        {escape(prompt[:180])}
                        {'...' if len(prompt) > 180 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    "🔓 Unpin",
                    key=f"unpin_{index}",
                    use_container_width=True
                ):

                    try:

                        prefix_ids = generator.tokenizer(
                            prompt,
                            add_special_tokens=False
                        ).input_ids

                        success = generator.unpin_prefix(prefix_ids)

                        if success:

                            st.session_state.pinned_prompts.remove(prompt)
                            st.success("Prefix released.")

                        else:

                            st.warning("Prefix was not found as pinned.")

                        st.rerun()

                    except Exception as error:

                        st.error(f"Unpin failed: {error}")


# ============================================================
# TAB 3: BENCHMARK ARENA
# ============================================================
with tab_arena:

    st.markdown(
        '<div class="section-header">BENCHMARK ARENA</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.generation_history

    cached_lat = list(
        st.session_state.latency_log["cached"]
    )

    uncached_lat = list(
        st.session_state.latency_log["uncached"]
    )

    if not history:

        st.markdown("""
        <div class="glass-panel" style="text-align:center;color:#64748b;">
            Run inferences in the Inference Lab to populate the arena.
        </div>
        """, unsafe_allow_html=True)

    else:

        c_mean, c_p50, c_p95 = safe_stats(cached_lat)
        u_mean, u_p50, u_p95 = safe_stats(uncached_lat)

        speedup = (
            u_mean / c_mean
            if c_mean > 0 and u_mean > 0 else 0
        )

        reduction = (
            (u_mean - c_mean) / u_mean * 100
            if u_mean > 0 else 0
        )

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            render_metric_card("AVG SPEEDUP", f"{speedup:.2f}×")

        with r2:
            render_metric_card(
                "LATENCY REDUCTION",
                f"{reduction:.1f}%"
            )

        with r3:
            render_metric_card("CACHED P50", f"{c_p50:.1f} ms")

        with r4:
            render_metric_card("UNCACHED P50", f"{u_p50:.1f} ms")

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:

            st.markdown("#### Cached Path")

            st.markdown(f"""
            <div class="glass-panel">
                <p>Mean: <b style="color:#00f0ff">{c_mean:.1f} ms</b></p>
                <p>P50 (Median): <b style="color:#00f0ff">{c_p50:.1f} ms</b></p>
                <p>P95: <b style="color:#00f0ff">{c_p95:.1f} ms</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:

            st.markdown("#### Uncached Path")

            st.markdown(f"""
            <div class="glass-panel">
                <p>Mean: <b style="color:#f87171">{u_mean:.1f} ms</b></p>
                <p>P50 (Median): <b style="color:#f87171">{u_p50:.1f} ms</b></p>
                <p>P95: <b style="color:#f87171">{u_p95:.1f} ms</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Latency History")

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            history_df,
            x="Mode",
            y="Latency (ms)"
        )

        if st.button(
            "🗑️ Clear Arena Data",
            use_container_width=True
        ):

            reset_arena()
            st.rerun()


# ============================================================
# TAB 4: GPU MONITOR
# ============================================================
with tab_gpu:

    st.markdown(
        '<div class="section-header">GPU MONITOR</div>',
        unsafe_allow_html=True
    )

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated() / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved() / (1024 ** 2)
        )

        total = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 2)
        )

        free = max(0.0, total - reserved)

        utilization = (
            reserved / total * 100
            if total else 0
        )

        g1, g2, g3, g4 = st.columns(4)

        with g1:
            render_metric_card("ALLOCATED", f"{allocated:.0f} MB")

        with g2:
            render_metric_card("RESERVED", f"{reserved:.0f} MB")

        with g3:
            render_metric_card("FREE", f"{free:.0f} MB")

        with g4:
            render_metric_card(
                "RESERVED VRAM",
                f"{utilization:.1f}%"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-panel">
            <div style="color:#a855f7;font-family:Orbitron;
                        margin-bottom:12px;">
                DEVICE INFO
            </div>

            <p>GPU: <b>{escape(torch.cuda.get_device_name(0))}</b></p>
            <p>TOTAL VRAM: <b>{total:.0f} MB</b></p>
            <p>CUDA: <b>{torch.version.cuda or "N/A"}</b></p>
            <p>PYTORCH: <b>{torch.__version__}</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### VRAM Reserved")

        st.progress(
            min(max(utilization / 100, 0.0), 1.0)
        )

        st.caption(
            f"{reserved:.0f} MB reserved / {total:.0f} MB total. "
            "This is PyTorch reserved memory, not a direct GPU "
            "compute-utilization reading."
        )

    else:

        st.markdown("""
        <div class="glass-panel" style="text-align:center;">
            <div style="color:#f87171;font-family:Orbitron;
                        margin-bottom:8px;">
                NO CUDA DEVICE
            </div>
            <div style="color:#64748b;">
                Running on CPU — GPU metrics unavailable.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#475569;font-size:.75rem;
            letter-spacing:1px;font-family:Orbitron;">
    PREFIX KV CACHE • EXPERIMENTAL NEURAL CORE •
    MEASUREMENTS VARY BY HARDWARE
</div>
""", unsafe_allow_html=True)
