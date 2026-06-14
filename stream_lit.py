"""Streamlit frontend for UIGPT.

UI architecture
---------------
1. **Streaming.** ``st.chat_input`` submits a query and the frontend
   manually iterates ``bridge.stream_message`` so it can route the two
   kinds of stream events (real assistant tokens vs. orchestrator
   "processing" progress strings) to different UI containers.
2. **Always-visible sidebar.** Streamlit's native collapse (which fully
   hides the panel) is disabled. Instead the sidebar is forced to always
   render and toggles between two widths - expanded 280px and collapsed
   64px - driven by ``html.sidebar-collapsed`` on the document element.
3. **Custom edge toggle.** A small arrow button is fixed to the
   sidebar's right edge. JS flips ``html.sidebar-collapsed`` and
   persists to localStorage; CSS transitions move the sidebar and the
   button together. Arrow points LEFT when expanded (click to shrink),
   RIGHT when collapsed (click to expand).
4. **Client-side theme.** Both palettes ship in one stylesheet, scoped
   to ``html.dark`` / ``html.light`` (with the dark palette also set on
   ``:root`` as a default so the page is fully styled even before JS
   attaches a class). The theme toggle is pure JS, so flipping
   mid-stream never interrupts streaming.
5. **Sidebar UIGPT heading.** A prominent brand mark sits above
    ``+ New Chat`` in the sidebar. When the sidebar collapses, only the
    circular icon remains visible.
6. **Persistent processing log.** Each assistant reply's intermediate
   "[Context Agent] Processing..." style strings are archived in
   ``st.session_state.processing_logs`` keyed by ``(thread_id, msg_idx)``
   so they survive ``st.rerun()`` and re-render in a collapsible
   ``st.expander`` next to the message on every subsequent visit.
"""

from __future__ import annotations

import ctypes
import threading
import uuid

import streamlit as st
import streamlit.components.v1 as components

import bridge


st.set_page_config(
    page_title="UIGPT - Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 1. Session state.
# ---------------------------------------------------------------------------
if "threads" not in st.session_state or "active_thread" not in st.session_state:
    _init = bridge.initialize_session()
    st.session_state.active_thread = _init["active_thread"]
    st.session_state.threads = _init["threads"]

if "processing_logs" not in st.session_state:
    st.session_state.processing_logs = {}

if "thread_messages" not in st.session_state:
    st.session_state.thread_messages = {
        st.session_state.active_thread: []
    }

if "active_run_id" not in st.session_state:
    st.session_state.active_run_id = None


class _RunCancelled(BaseException):
    """Internal control-flow signal for user-requested termination."""


def _raise_thread_exception(thread_id: int | None, exc_type: type[BaseException]) -> bool:
    """Ask CPython to raise ``exc_type`` in the target worker thread.

    The worker is dedicated to one background graph stream, so this is
    used only for explicit user termination. If the thread is currently
    blocked inside a provider call, delivery still depends on the next
    interpreter checkpoint.
    """
    if not thread_id:
        return False

    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(exc_type),
    )
    if result == 0:
        return False
    if result > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
        return False
    return True


def _get_label_llm():
    """Reuse an already-initialised cheap LLM from the backend."""
    from fyp import _schema_model
    return _schema_model


@st.cache_resource
def _get_backend_warmup_state() -> dict[str, object]:
    """Persist one backend warm-up worker across reruns."""
    return {
        "lock": threading.Lock(),
        "started": False,
        "done": False,
        "error": None,
        "worker": None,
    }


def _ensure_backend_warmup_started() -> None:
    """Preload the heavy backend in the background after the UI shell renders."""
    state = _get_backend_warmup_state()
    with state["lock"]:
        if state["started"]:
            return
        state["started"] = True

        def _warm_backend() -> None:
            try:
                bridge._get_graph()
            except Exception as exc:  # pragma: no cover - best effort only
                with state["lock"]:
                    state["error"] = exc
                    state["done"] = True
                return

            with state["lock"]:
                state["done"] = True

        worker = threading.Thread(
            target=_warm_backend,
            name="uigpt-backend-warmup",
            daemon=True,
        )
        state["worker"] = worker

    worker.start()


@st.cache_resource
def _get_background_run_store() -> dict[str, object]:
    """Persist background worker state across Streamlit reruns."""
    return {
        "lock": threading.Lock(),
        "runs": {},
    }


def _get_thread_messages(thread_id: str, *, refresh: bool = False) -> list[dict[str, str]]:
    """Return cached display messages for a thread, loading once on demand.

    The sidebar already renders purely from ``st.session_state``. By
    caching the main-panel messages alongside it, we avoid a synchronous
    ``graph.get_state(...)`` call on every rerun and only hit the backend
    when a thread is first opened or explicitly refreshed after a stream.
    """
    cache = st.session_state.thread_messages
    if refresh or thread_id not in cache:
        cache[thread_id] = bridge.switch_chat(thread_id)
    return cache.get(thread_id, [])


def _get_run_snapshot(run_id: str | None) -> dict[str, object] | None:
    """Return a detached snapshot of one background run."""
    if not run_id:
        return None
    store = _get_background_run_store()
    with store["lock"]:
        run = store["runs"].get(run_id)
        if not run:
            return None
        return {
            "id": run["id"],
            "thread_id": run["thread_id"],
            "assistant_idx": run["assistant_idx"],
            "response": run["response"],
            "processing_steps": list(run["processing_steps"]),
            "status": run["status"],
            "threads": dict(run["threads"]),
            "error": run["error"],
        }


def _discard_run(run_id: str | None) -> None:
    if not run_id:
        return
    store = _get_background_run_store()
    with store["lock"]:
        store["runs"].pop(run_id, None)


def _cancel_background_run(run_id: str | None) -> None:
    """Terminate one active background run as soon as the worker can stop."""
    if not run_id:
        return

    store = _get_background_run_store()
    cancel_event = None
    worker = None
    with store["lock"]:
        state = store["runs"].pop(run_id, None)
        if not state:
            return
        cancel_event = state.get("cancel_event")
        worker = state.get("worker")

    if isinstance(cancel_event, threading.Event):
        cancel_event.set()
    if isinstance(worker, threading.Thread) and worker.is_alive():
        _raise_thread_exception(worker.ident, _RunCancelled)


def _start_background_run(
    *,
    query: str,
    thread_id: str,
    assistant_idx: int,
    threads: dict[str, str],
) -> str:
    """Consume one StreamSession in a daemon thread.

    Streamlit reruns must stay cheap so RECENT buttons remain usable
    while a response is being generated. The worker owns the blocking
    stream iteration; the UI just polls shared state and re-renders.
    """
    run_id = str(uuid.uuid4())
    label_llm = _get_label_llm()
    threads_snapshot = dict(threads)
    store = _get_background_run_store()
    cancel_event = threading.Event()

    with store["lock"]:
        store["runs"][run_id] = {
            "id": run_id,
            "thread_id": thread_id,
            "assistant_idx": assistant_idx,
            "response": "",
            "processing_steps": [],
            "status": "running",
            "threads": dict(threads_snapshot),
            "error": None,
            "cancel_event": cancel_event,
            "worker": None,
        }

    def worker() -> None:
        session = bridge.stream_message(
            query=query,
            thread_id=thread_id,
            threads=threads_snapshot,
            llm=label_llm,
        )
        try:
            for kind, text in session.stream_events():
                if cancel_event.is_set():
                    raise _RunCancelled()

                with store["lock"]:
                    state = store["runs"].get(run_id)
                    if not state or cancel_event.is_set():
                        raise _RunCancelled()
                    if kind == "answer":
                        state["response"] = str(state["response"]) + text
                    elif kind == "processing":
                        state["processing_steps"] = list(session.processing_steps)

            if cancel_event.is_set():
                raise _RunCancelled()

            with store["lock"]:
                state = store["runs"].get(run_id)
                if not state or cancel_event.is_set():
                    raise _RunCancelled()
                state["response"] = session.response
                state["processing_steps"] = list(session.processing_steps)
                state["threads"] = dict(session.threads)
                state["status"] = "complete"
        except _RunCancelled:
            return
        except Exception as exc:
            with store["lock"]:
                state = store["runs"].get(run_id)
                if not state:
                    return
                state["status"] = "error"
                state["error"] = f"{type(exc).__name__}: {exc}"

    worker_thread = threading.Thread(
        target=worker,
        name=f"uigpt-stream-{run_id}",
        daemon=True,
    )
    with store["lock"]:
        state = store["runs"].get(run_id)
        if state:
            state["worker"] = worker_thread
    worker_thread.start()
    return run_id


def _sync_active_run() -> dict[str, object] | None:
    """Project the current background run into session-state caches."""
    run_id = st.session_state.active_run_id
    snapshot = _get_run_snapshot(run_id)
    if not snapshot:
        st.session_state.active_run_id = None
        return None

    thread_id = str(snapshot["thread_id"])
    assistant_idx = int(snapshot["assistant_idx"])
    cache = st.session_state.thread_messages.setdefault(thread_id, [])

    while len(cache) <= assistant_idx:
        cache.append({"role": "assistant", "content": ""})

    cache[assistant_idx]["content"] = str(snapshot["response"] or "")

    processing_steps = list(snapshot["processing_steps"])
    if processing_steps:
        st.session_state.processing_logs[(thread_id, assistant_idx)] = processing_steps

    status = snapshot["status"]
    if status == "complete":
        st.session_state.threads = dict(snapshot["threads"])
        st.session_state.active_run_id = None
        _discard_run(run_id)
    elif status == "error":
        cache[assistant_idx]["content"] = str(snapshot["error"] or "Generation failed.")
        st.session_state.active_run_id = None
        _discard_run(run_id)

    return snapshot


def _set_stream_interaction_lock(locked: bool) -> None:
    """Clear any stale client-side sidebar lock from older UI logic.

    Responses now stream from a background worker that survives reruns,
    so sidebar navigation must stay available while a reply is in
    flight. We therefore never apply the old sidebar lock and only use
    this bridge to make sure a previously-set lock class is removed.
    """
    _ = locked
    locked_js = "false"
    components.html(
        f"""
        <script>
        (function() {{
            try {{
                var pwin = window.parent;
                if (typeof pwin.setUigptStreamLock === 'function') {{
                    pwin.setUigptStreamLock({locked_js});
                }}
            }} catch (e) {{
                console.error('UIGPT: failed to toggle stream lock', e);
            }}
        }})();
        </script>
        """,
        height=0,
    )


# ---------------------------------------------------------------------------
# 2. Pre-flight JS via components.html (st.markdown does NOT execute scripts).
# ---------------------------------------------------------------------------
components.html(
    """
    <script>
    (function() {
        try {
            var pdoc = window.parent.document;
            var pwin = window.parent;
            var html = pdoc.documentElement;
            var t = null;
            try { t = pwin.localStorage.getItem('uigpt-theme'); } catch (e) {}
            if (t !== 'light' && t !== 'dark') t = 'dark';
            html.classList.remove('dark', 'light');
            html.classList.add(t);
            try {
                if (pwin.localStorage.getItem('uigpt-sidebar') === 'collapsed') {
                    html.classList.add('sidebar-collapsed');
                } else {
                    html.classList.remove('sidebar-collapsed');
                }
            } catch (e) {}
            pwin.toggleUigptTheme = function() {
                var h = pwin.document.documentElement;
                var next = h.classList.contains('dark') ? 'light' : 'dark';
                h.classList.remove('dark', 'light');
                h.classList.add(next);
                try { pwin.localStorage.setItem('uigpt-theme', next); } catch (e) {}
            };
            pwin.toggleUigptSidebar = function() {
                var h = pwin.document.documentElement;
                h.classList.toggle('sidebar-collapsed');
                try {
                    pwin.localStorage.setItem(
                        'uigpt-sidebar',
                        h.classList.contains('sidebar-collapsed') ? 'collapsed' : 'open'
                    );
                } catch (e) {}
            };
            pwin.setUigptStreamLock = function(locked) {
                html.classList.toggle('uigpt-stream-locked', !!locked);
            };
            pwin.setUigptStreamLock(false);

            var scrollStateKey = 'uigpt-scroll-state:v1';
            var getScrollMetrics = function() {
                var docEl = pdoc.documentElement;
                var body = pdoc.body;
                var scrollY = pwin.scrollY || docEl.scrollTop || body.scrollTop || 0;
                var viewportHeight = pwin.innerHeight || docEl.clientHeight || 0;
                var documentHeight = Math.max(
                    body.scrollHeight || 0,
                    docEl.scrollHeight || 0,
                    body.offsetHeight || 0,
                    docEl.offsetHeight || 0,
                    body.clientHeight || 0,
                    docEl.clientHeight || 0
                );
                return {
                    scrollY: scrollY,
                    viewportHeight: viewportHeight,
                    documentHeight: documentHeight,
                    distanceFromBottom: Math.max(0, documentHeight - (scrollY + viewportHeight))
                };
            };
            var persistScrollState = function() {
                try {
                    var metrics = getScrollMetrics();
                    pwin.sessionStorage.setItem(scrollStateKey, JSON.stringify({
                        scrollY: metrics.scrollY,
                        stickToBottom: metrics.distanceFromBottom < 160
                    }));
                } catch (e) {}
            };
            var restoreScrollState = function() {
                try {
                    var raw = pwin.sessionStorage.getItem(scrollStateKey);
                    if (!raw) return;
                    var saved = JSON.parse(raw);
                    if (!saved || typeof saved.scrollY !== 'number') return;
                    var apply = function() {
                        var metrics = getScrollMetrics();
                        var maxScrollY = Math.max(0, metrics.documentHeight - metrics.viewportHeight);
                        var targetY = saved.stickToBottom
                            ? maxScrollY
                            : Math.min(Math.max(0, saved.scrollY), maxScrollY);
                        pwin.scrollTo(0, targetY);
                    };
                    if (Array.isArray(pwin.__uigptScrollRestoreTimers)) {
                        for (var i = 0; i < pwin.__uigptScrollRestoreTimers.length; i++) {
                            pwin.clearTimeout(pwin.__uigptScrollRestoreTimers[i]);
                        }
                    }
                    var applySoon = function() {
                        pwin.requestAnimationFrame(function() {
                            pwin.requestAnimationFrame(apply);
                        });
                    };
                    pwin.__uigptScrollRestoreTimers = [
                        pwin.setTimeout(applySoon, 0),
                        pwin.setTimeout(apply, 80),
                        pwin.setTimeout(apply, 220),
                        pwin.setTimeout(apply, 500)
                    ];
                } catch (e) {}
            };
            restoreScrollState();
            if (!pwin.__uigptScrollStateBound) {
                var scrollTicking = false;
                var scheduleScrollPersist = function() {
                    if (scrollTicking) return;
                    scrollTicking = true;
                    pwin.requestAnimationFrame(function() {
                        scrollTicking = false;
                        persistScrollState();
                    });
                };
                pwin.addEventListener('scroll', scheduleScrollPersist, { passive: true });
                pwin.addEventListener('beforeunload', persistScrollState);
                pwin.document.addEventListener('click', function(evt) {
                    var target = evt.target;
                    if (!target || !target.closest) return;
                    if (target.closest('button, [role="button"], a[href]')) {
                        persistScrollState();
                    }
                }, true);
                pwin.document.addEventListener('submit', persistScrollState, true);
                pwin.document.addEventListener('keydown', function(evt) {
                    if (evt.key !== 'Enter' || evt.shiftKey) return;
                    var target = evt.target;
                    if (!target || !target.closest) return;
                    if (target.closest('[data-testid="stChatInput"]')) {
                        persistScrollState();
                    }
                }, true);
                pwin.__uigptScrollStateBound = true;
            }

            // Reparent the sidebar/theme toggles to <body> so that no
            // ancestor with `transform`/`filter`/`will-change` can
            // change their containing block (those properties make a
            // `position: fixed` element position relative to the
            // ancestor instead of the viewport — Streamlit applies
            // some of these on its main wrapper). This also flushes
            // them OUT of any local stacking context so their huge
            // z-index actually wins against the sidebar.
            //
            // IMPORTANT: Streamlit re-renders the markdown on every
            // rerun (e.g. clicking "+ New Chat" or a thread button),
            // which injects FRESH copies of the toggle elements into
            // the markdown container. Without cleanup the old copies
            // pile up inside <body>, and the constant DOM-mutation
            // polling that used to live here was disrupting the
            // <textarea> inside st.chat_input — every appendChild
            // triggered React reconciliation and stole focus from
            // the input as the user typed. We now:
            //   1. Reconcile duplicates idempotently in moveToggle
            //      (keep one bound copy in body, delete the rest).
            //   2. Use a MutationObserver instead of setInterval so
            //      the function only runs when the DOM ACTUALLY
            //      changes (not on a 500ms timer that fires while
            //      the user is mid-keystroke).
            var bindSidebarToggle = function(el) {
                if (!el) return;
                el.onclick = function() {
                    if (typeof pwin.toggleUigptSidebar === 'function') {
                        pwin.toggleUigptSidebar();
                    }
                };
                el.dataset.uigptBound = '1';
            };
            var bindThemeToggle = function(wrap) {
                if (!wrap) return;
                // onclick attributes are stripped by DOMPurify even
                // with unsafe_allow_html=True, so the inline
                // onclick="window.toggleUigptTheme()" never fires.
                // Attach the listener here on the inner <button>.
                var b = wrap.querySelector('button');
                if (b) {
                    b.onclick = function() {
                        if (typeof pwin.toggleUigptTheme === 'function') {
                            pwin.toggleUigptTheme();
                        }
                    };
                }
                wrap.dataset.uigptBound = '1';
            };
            var reconcile = function(selector, binder) {
                var nodes = pdoc.querySelectorAll(selector);
                if (!nodes.length) return;
                // Prefer the already-bound copy that lives in <body>.
                var keep = null;
                for (var i = 0; i < nodes.length; i++) {
                    if (nodes[i].parentElement === pdoc.body && nodes[i].dataset.uigptBound) {
                        keep = nodes[i];
                        break;
                    }
                }
                if (!keep) {
                    // Otherwise promote the first one to body and bind it.
                    keep = nodes[0];
                    if (keep.parentElement !== pdoc.body) {
                        pdoc.body.appendChild(keep);
                    }
                }
                // Always rebind the kept copy. Streamlit reruns can
                // leave us with a visually preserved node whose prior
                // JS listener is gone, so binding cannot be treated as
                // a one-time operation.
                binder(keep);
                // Remove any other copies so we don't accumulate
                // overlapping fixed-position duplicates on rerun.
                for (var j = 0; j < nodes.length; j++) {
                    if (nodes[j] !== keep && nodes[j].parentNode) {
                        nodes[j].parentNode.removeChild(nodes[j]);
                    }
                }
            };
            var moveToggle = function() {
                reconcile('.uigpt-sidebar-toggle', bindSidebarToggle);
                reconcile('.uigpt-floating-toggle', bindThemeToggle);
            };
            moveToggle();
            // Replace the old setInterval polling with a one-shot
            // MutationObserver scoped to <body>. It fires only when
            // Streamlit actually adds new toggle nodes — never while
            // the user is typing — so the chat input's focus is no
            // longer stolen mid-keystroke.
            if (pwin.__uigptToggleInterval) {
                try { pwin.clearInterval(pwin.__uigptToggleInterval); } catch (e) {}
                pwin.__uigptToggleInterval = null;
            }
            if (pwin.__uigptToggleObserver) {
                try { pwin.__uigptToggleObserver.disconnect(); } catch (e) {}
                pwin.__uigptToggleObserver = null;
            }
            try {
                var obs = new pwin.MutationObserver(function(muts) {
                    for (var i = 0; i < muts.length; i++) {
                        var added = muts[i].addedNodes;
                        for (var j = 0; j < added.length; j++) {
                            var n = added[j];
                            if (n && n.nodeType === 1 && (
                                (n.classList && (n.classList.contains('uigpt-sidebar-toggle') || n.classList.contains('uigpt-floating-toggle'))) ||
                                (n.querySelector && (n.querySelector('.uigpt-sidebar-toggle') || n.querySelector('.uigpt-floating-toggle')))
                            )) {
                                moveToggle();
                                return;
                            }
                        }
                    }
                });
                obs.observe(pdoc.body, { childList: true, subtree: true });
                pwin.__uigptToggleObserver = obs;
            } catch (e) {
                // MutationObserver unavailable — fall back to a much
                // slower poll so we still survive in odd environments
                // without thrashing the DOM mid-typing.
                pwin.__uigptToggleInterval = pwin.setInterval(moveToggle, 2000);
            }
        } catch (e) {
            console.error('UIGPT: theme/sidebar JS failed to attach', e);
        }
    })();
    </script>
    """,
    height=0,
)


# ---------------------------------------------------------------------------
# 3. Static stylesheet.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap");
    @import url("https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap");

    :root, html.dark {
        --bg-app:        #060e20;
        --bg-main:       #0f1a34;
        --bg-panel:      #0b1326;
        --bg-card:       #171f33;
        --bg-card-hi:    #222a3d;
        --bg-input:      #1e293b;
        --text-primary:  #f1f5f9;
        --text-secondary:#dae2fd;
        --text-muted:    #94a3b8;
        --accent:        #38bdf8;
        --accent-hi:     #7bd0ff;
        --border:        rgba(255, 255, 255, 0.08);
        --border-hi:     rgba(56, 189, 248, 0.45);
        --user-bubble:   #222a3d;
        --user-text:     #f1f5f9;
        --assistant-bubble: #171f33;
    }
    html.light {
        --bg-app:        #f8fafc;
        --bg-main:       #ffffff;
        --bg-panel:      #ffffff;
        --bg-card:       #ffffff;
        --bg-card-hi:    #f1f5f9;
        --bg-input:      #ffffff;
        --text-primary:  #0f172a;
        --text-secondary:#1e293b;
        --text-muted:    #64748b;
        --accent:        #0284c7;
        --accent-hi:     #0369a1;
        --border:        rgba(15, 23, 42, 0.10);
        --border-hi:     rgba(2, 132, 199, 0.45);
        --user-bubble:   #e0f2fe;
        --user-text:     #0c4a6e;
        --assistant-bubble: #ffffff;
    }

    html { --uigpt-sidebar-w: 280px; }
    html.sidebar-collapsed { --uigpt-sidebar-w: 64px; }

    html, body, .stApp {
        background: var(--bg-app) !important;
        color: var(--text-primary) !important;
        font-family: "Manrope", sans-serif !important;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: "Manrope", sans-serif !important;
    }
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: "Manrope", sans-serif !important;
    }

    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    #MainMenu,
    footer { display: none !important; }
    header[data-testid="stHeader"] {
        background: var(--bg-main) !important;
        height: auto !important;
        z-index: 1000 !important;
    }

    /* The sidebar's z-index must be defined and LOW enough that our
       fixed toggle button can paint on top of it. Streamlit's default
       can be in the hundreds-of-thousands range, which is why the
       button used to disappear behind the sidebar. We pin it to 100
       and ensure the toggle uses a much higher z-index below. */
    section[data-testid="stSidebar"] {
        position: relative !important;
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        width: var(--uigpt-sidebar-w) !important;
        min-width: var(--uigpt-sidebar-w) !important;
        max-width: var(--uigpt-sidebar-w) !important;
        flex: 0 0 var(--uigpt-sidebar-w) !important;
        background: var(--bg-panel) !important;
        border-right: 1px solid var(--border) !important;
        /* Slimmer shadow so it doesn't visually swallow the toggle. */
        box-shadow: 2px 0 6px rgba(0, 0, 0, 0.12) !important;
        transition: width 0.25s ease, min-width 0.25s ease,
                    max-width 0.25s ease, flex-basis 0.25s ease !important;
        overflow: hidden !important;
        text-align: left !important;
        z-index: 100 !important;
    }
    /* Newer Streamlit ships nested content containers
       (stSidebarContent, stSidebarUserContent). Without these
       overrides the inner div keeps its default 280px width and the
       outer section can't actually shrink. */
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        background: var(--bg-panel) !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
        margin-top: -16px !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: var(--text-secondary) !important;
        font-family: "Inter", "Manrope", sans-serif !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Outlined" !important;
        font-size: 20px !important;
        line-height: 1 !important;
        font-variation-settings: 'FILL' 0, 'wght' 500, 'GRAD' 0, 'opsz' 24;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        text-align: left !important;
        font-family: "Inter", "Manrope", sans-serif !important;
        font-weight: 500 !important;
        padding: 0.55rem 0.8rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        transition: all 0.18s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button > div,
    section[data-testid="stSidebar"] .stButton > button span[data-has-shortcut="false"],
    section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {
        text-align: left !important;
    }
    section[data-testid="stSidebar"] .stButton > button:not(:has([data-testid="stIconMaterial"])) {
        border-color: var(--bg-card) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:not(:has([data-testid="stIconMaterial"])) > div,
    section[data-testid="stSidebar"] .stButton > button:not(:has([data-testid="stIconMaterial"])) span[data-has-shortcut="false"] {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] .stButton > button:not(:has([data-testid="stIconMaterial"])) [data-testid="stMarkdownContainer"] {
        text-align: left !important;
    }
    section[data-testid="stSidebar"] .stButton {
        margin: 0 0 10px !important;
    }
    html.uigpt-stream-locked section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]) {
        pointer-events: auto !important;
        cursor: pointer !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]) {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]) > div {
        display: flex !important;
        align-items: center !important;
        gap: 0.55rem !important;
    }
    section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]) [data-testid="stMarkdownContainer"] {
        min-width: 0 !important;
        flex: 1 1 auto !important;
        overflow: hidden !important;
        color: #64748b !important;
        font-family: "Manrope", sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        line-height: 20px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]) [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        color: inherit !important;
        font: inherit !important;
        line-height: inherit !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-card-hi) !important;
        border-color: var(--border-hi) !important;
        color: var(--accent-hi) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]):hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: var(--text-primary) !important;
        border-color: transparent !important;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08),
                    0 0 0 3px rgba(56, 189, 248, 0.10) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]):hover [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:has([data-testid="stIconMaterial"]):hover [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--bg-card-hi) !important;
        color: var(--accent-hi) !important;
        border-color: var(--border-hi) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:has([data-testid="stIconMaterial"]) {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:has([data-testid="stIconMaterial"]):hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: var(--text-primary) !important;
        border-color: transparent !important;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08),
                    0 0 0 3px rgba(56, 189, 248, 0.10) !important;
    }

    html.sidebar-collapsed section[data-testid="stSidebar"] .stButton > button {
        width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    html.sidebar-collapsed section[data-testid="stSidebar"] .stButton,
    html.sidebar-collapsed section[data-testid="stSidebar"] .stButton > div {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    html.sidebar-collapsed section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    html.sidebar-collapsed section[data-testid="stSidebar"] .stButton > button > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 100% !important;
        overflow: hidden !important;
    }
    html.sidebar-collapsed section[data-testid="stSidebar"] .stButton > button p {
        max-width: 1.2em !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-overflow: clip !important;
        margin: 0 !important;
        font-size: 18px !important;
        line-height: 1 !important;
    }
    html.sidebar-collapsed section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]) > div {
        gap: 0 !important;
    }
    html.sidebar-collapsed section[data-testid="stSidebar"] .stButton > button:has([data-testid="stIconMaterial"]) [data-testid="stMarkdownContainer"] {
        display: none !important;
    }
    html.sidebar-collapsed .uigpt-brand .name,
    html.sidebar-collapsed .uigpt-recent-label {
        display: none !important;
    }
    html.sidebar-collapsed .uigpt-brand {
        justify-content: center !important;
    }

    /* Sidebar edge toggle — anchored to the right edge of the sidebar
       via calc(). It tracks --uigpt-sidebar-w automatically so we
       don't need a separate "collapsed" override for `left`.

       z-index is intentionally astronomical so we always paint over
       Streamlit's sidebar (which carries an internal z-index in the
       hundreds-of-thousands range — that's why this button used to
       disappear behind the panel). */
    .uigpt-sidebar-toggle {
        position: fixed;
        top: 22px;
        z-index: 2147483000;  /* basically max int32 — wins all stacking */
        width: 30px;
        height: 30px;
        padding: 0;
        border: 1px solid var(--border-hi);
        border-radius: 50%;
        background: var(--bg-card);
        color: transparent;
        cursor: pointer;
        outline: none;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.28);
        transition: left 0.25s ease, background 0.15s ease,
                    transform 0.15s ease;
        /* Center on the sidebar's right edge: half-on, half-off.
           14px = half the button's 30px width minus 1px border. */
        left: calc(var(--uigpt-sidebar-w) - 15px);
        display: block;
    }
    .uigpt-sidebar-toggle:hover {
        background: var(--bg-card-hi);
        transform: scale(1.06);
    }
    /* Default (expanded) → arrow points LEFT (click to shrink). */
    .uigpt-sidebar-toggle::after {
        content: "";
        position: absolute;
        top: 50%; left: 50%;
        width: 9px; height: 9px;
        border-left: 2px solid var(--text-primary);
        border-bottom: 2px solid var(--text-primary);
        border-right: 0; border-top: 0;
        transform: translate(-30%, -50%) rotate(45deg);
        transition: transform 0.2s ease;
    }
    /* Collapsed → arrow points RIGHT (click to expand). */
    html.sidebar-collapsed .uigpt-sidebar-toggle::after {
        border-left: 0; border-bottom: 0;
        border-right: 2px solid var(--text-primary);
        border-top: 2px solid var(--text-primary);
        transform: translate(-70%, -50%) rotate(45deg);
    }

    div[data-testid="stChatInput"],
    div[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"] {
        background: var(--bg-main) !important;
    }
    div[data-testid="stTextInputRootElement"] > div {
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInputRootElement"] input {
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
        caret-color: var(--text-primary) !important;
    }
    div[data-testid="stTextInputRootElement"] input::placeholder {
        color: var(--text-muted) !important;
    }
    /* Streamlit's default z-index for the sticky bottom bar is
       sidebar-1 (=99), which sits UNDER our overridden sidebar
       (z-index:100). Although they don't normally overlap as flex
       siblings, we lift the chat input above any stray fixed/sticky
       wrappers so clicks always land on the textarea. We also pin
       pointer-events: auto in case any ancestor forced 'none'. */
    div[data-testid="stBottom"] {
        left: var(--uigpt-sidebar-w) !important;
        width: calc(100vw - var(--uigpt-sidebar-w)) !important;
        z-index: 200 !important;
        pointer-events: auto !important;
    }
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] textarea {
        pointer-events: auto !important;
    }
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div,
    div[data-testid="stChatInput"] [data-baseweb],
    div[data-testid="stChatInput"] [data-baseweb="textarea"],
    div[data-testid="stChatInput"] [data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] > div > div:first-child {
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] textarea {
        background: var(--bg-input) !important;
        color: var(--text-primary) !important;
        caret-color: var(--text-primary) !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
    }
    button[data-testid="stChatInputSubmitButton"],
    div[data-testid="stChatInput"] button {
        width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #7dd3fc !important;
        color: #05070b !important;
        border: 1px solid rgba(125, 211, 252, 0.55) !important;
        border-radius: 12px !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.18) !important;
        cursor: pointer !important;
        transition: background 0.15s ease, border-color 0.15s ease,
                    color 0.15s ease, box-shadow 0.15s ease,
                    transform 0.15s ease !important;
    }
    button[data-testid="stChatInputSubmitButton"]:hover,
    div[data-testid="stChatInput"] button:hover {
        background: #a5e5ff !important;
        border-color: #7dd3fc !important;
        color: #05070b !important;
        box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.28) !important;
        transform: translateY(-1px);
    }
    button[data-testid="stChatInputSubmitButton"] svg,
    button[data-testid="stChatInputSubmitButton"] svg *,
    div[data-testid="stChatInput"] button svg,
    div[data-testid="stChatInput"] button svg * {
        display: none !important;
    }
    button[data-testid="stChatInputSubmitButton"]::before,
    div[data-testid="stChatInput"] button::before {
        content: "↑";
        font-size: 18px;
        font-weight: 700;
        line-height: 1;
        color: currentColor;
    }
    [data-testid="stAppViewContainer"] .stButton > button[kind="secondary"] {
        min-height: 46px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(248, 113, 113, 0.45) !important;
        background: rgba(127, 29, 29, 0.28) !important;
        color: #fecaca !important;
        font-weight: 700 !important;
        box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.12) !important;
        transition: background 0.15s ease, border-color 0.15s ease,
                    box-shadow 0.15s ease, transform 0.15s ease !important;
    }
    [data-testid="stAppViewContainer"] .stButton > button[kind="secondary"]:hover {
        background: rgba(153, 27, 27, 0.34) !important;
        border-color: rgba(252, 165, 165, 0.72) !important;
        box-shadow: 0 0 0 4px rgba(248, 113, 113, 0.18) !important;
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        min-height: unset !important;
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transform: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: var(--bg-card-hi) !important;
        border-color: var(--border-hi) !important;
        color: var(--accent-hi) !important;
        box-shadow: none !important;
        transform: none !important;
    }

    div[data-testid="stChatMessage"] {
        background: transparent !important;
        color: var(--text-primary) !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.5rem 0 !important;
        display: flex !important;
        align-items: flex-start !important;
        gap: 12px !important;
    }
    div[data-testid="stChatMessage"] > div:not([data-testid*="Avatar"]):not([data-testid*="avatar"]) {
        background: var(--assistant-bubble) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--assistant-bubble) !important;
        border-radius: 14px !important;
        padding: 0.7rem 1rem !important;
        max-width: 80% !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row !important;
        justify-content: flex-start !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        > div:not([data-testid*="Avatar"]):not([data-testid*="avatar"]) {
        background: var(--assistant-bubble) !important;
        color: var(--text-primary) !important;
        border-color: var(--assistant-bubble) !important;
        margin-right: auto !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse !important;
        justify-content: flex-end !important;
        text-align: left !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        > div:not([data-testid*="Avatar"]):not([data-testid*="avatar"]) {
        background: var(--user-bubble) !important;
        color: var(--user-text) !important;
        border-color: var(--user-bubble) !important;
        margin-left: auto !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: #ffffff !important;
        flex-shrink: 0 !important;
    }
    div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] *,
    div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    div[data-testid="stChatMessage"] details,
    div[data-testid="stChatMessage"] [data-testid="stExpander"],
    div[data-testid="stChatMessage"] [data-testid="stStatusWidget"] {
        background: var(--bg-card-hi) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        margin-bottom: 0.5rem !important;
    }
    div[data-testid="stChatMessage"] details summary,
    div[data-testid="stChatMessage"] [data-testid="stExpander"] summary,
    div[data-testid="stChatMessage"] [data-testid="stStatusWidget"] summary {
        color: var(--text-secondary) !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.7rem !important;
    }

    .uigpt-floating-toggle {
        position: fixed;
        top: 12px;
        right: 16px;
        z-index: 60;
    }
    .uigpt-floating-toggle button {
        background: transparent;
        border: none;
        outline: none;
        box-shadow: none;
        color: var(--text-primary);
        width: 40px;
        height: 40px;
        border-radius: 50%;
        cursor: pointer;
        font-size: 20px;
        line-height: 1;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s ease, color 0.15s ease;
    }
    .uigpt-floating-toggle button:hover {
        background: var(--bg-card-hi);
        color: var(--accent);
    }
    html.dark  .uigpt-theme-icon::before { content: "☀"; }
    html.light .uigpt-theme-icon::before { content: "☾"; }

    .uigpt-welcome { text-align: center; padding: 96px 16px 48px; }
    .uigpt-welcome h1 {
        font-size: 32px; font-weight: 700;
        color: var(--text-primary); margin: 0 0 8px;
    }
    .uigpt-welcome p {
        color: var(--text-muted); font-size: 18px; margin: 0;
    }
    .uigpt-brand {
        display: flex; align-items: center; gap: 10px;
        min-height: 40px;
        margin: 0 0 12px;
        padding: 0 4px;
    }
    .uigpt-brand .cloud-icon {
        position: relative;
        width: 22px;
        height: 22px;
        flex: 0 0 22px;
        display: inline-block;
        border-radius: 50%;
        overflow: hidden;
        background:
            radial-gradient(circle at 28% 26%, rgba(255, 255, 255, 0.24) 0 10%, transparent 11%),
            radial-gradient(circle at 72% 30%, rgba(255, 255, 255, 0.18) 0 13%, transparent 14%),
            linear-gradient(180deg, #89d0ff 0%, #58b2f4 52%, #2f7fd3 100%);
        border: 1px solid rgba(255, 255, 255, 0.22);
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.18);
    }
    .uigpt-brand .cloud-icon::before,
    .uigpt-brand .cloud-icon::after {
        content: "";
        position: absolute;
        background: rgba(255, 255, 255, 0.92);
        border-radius: 999px;
    }
    .uigpt-brand .cloud-icon::before {
        width: 15px;
        height: 8px;
        left: 3px;
        bottom: 5px;
        box-shadow: 5px -3px 0 1px rgba(255, 255, 255, 0.94),
                    -2px -2px 0 0 rgba(255, 255, 255, 0.9);
    }
    .uigpt-brand .cloud-icon::after {
        width: 18px;
        height: 6px;
        left: 2px;
        bottom: 3px;
        background: rgba(255, 255, 255, 0.86);
    }
    .uigpt-brand .name {
        font-family: "Inter", "Manrope", sans-serif;
        font-weight: 700;
        font-size: 19px;
        letter-spacing: 0.08em;
        color: var(--text-primary);
    }
    .uigpt-recent-label {
        font-family: "Inter", "Manrope", sans-serif;
        margin: 4px 0 10px;
        padding: 0 4px; color: var(--text-muted);
        text-transform: uppercase; font-size: 11px;
        letter-spacing: 0.16em; font-weight: 400;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 4. Sidebar toggle + theme toggle.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <button class="uigpt-sidebar-toggle"
            type="button"
            onclick="window.toggleUigptSidebar()"
            aria-label="Toggle sidebar"></button>
    <div class="uigpt-floating-toggle">
        <button type="button" onclick="window.toggleUigptTheme()" aria-label="Toggle theme">
            <span class="uigpt-theme-icon"></span>
        </button>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 5. Sidebar.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="uigpt-brand" role="heading" aria-level="1">
            <span class="cloud-icon" aria-hidden="true"></span>
            <span class="name">UIGPT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋ New Chat", key="btn_new_chat", use_container_width=True):
        result = bridge.new_chat(st.session_state.threads)
        st.session_state.active_thread = result["active_thread"]
        st.session_state.threads = result["threads"]
        st.session_state.thread_messages[result["active_thread"]] = []
        st.rerun()

    st.markdown(
        '<div class="uigpt-recent-label">Recent</div>',
        unsafe_allow_html=True,
    )

    for tid, label in list(st.session_state.threads.items()):
        is_active = tid == st.session_state.active_thread
        display_label = f"{label or 'New Chat'}"
        if st.button(
            display_label,
            key=f"thread_{tid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            icon=":material/history:",
        ):
            _get_thread_messages(tid)
            st.session_state.active_thread = tid
            st.rerun()


# ---------------------------------------------------------------------------
# 6. Main chat area - history with persistent processing-step expanders.
# ---------------------------------------------------------------------------
_chat_panel_refresh_interval = "250ms" if st.session_state.active_run_id else None

_ensure_backend_warmup_started()


@st.fragment(run_every=_chat_panel_refresh_interval)
def _render_chat_panel() -> None:
    active_run = _sync_active_run()
    has_running_generation = bool(active_run and active_run.get("status") == "running")
    active_run_thread_id = str(active_run["thread_id"]) if active_run else None
    is_current_thread_generating = bool(
        has_running_generation
        and active_run_thread_id == st.session_state.active_thread
    )

    _set_stream_interaction_lock(has_running_generation)

    messages = _get_thread_messages(st.session_state.active_thread)

    if not messages:
        st.markdown(
            """
            <div class="uigpt-welcome">
                <h1>How can I help you today?</h1>
                <p>Ask anything to get started — design research, layouts, code, or UI components.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for idx, msg in enumerate(messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    steps = st.session_state.processing_logs.get(
                        (st.session_state.active_thread, idx)
                    )
                    if steps:
                        with st.expander("Processing details", expanded=False):
                            for s in steps:
                                st.markdown(f"_{s}_")
                st.markdown(msg["content"])

    if has_running_generation:
        if active_run_thread_id != st.session_state.active_thread:
            st.caption(
                "A reply is still generating in another chat. You can switch "
                "threads without losing it, but you need to wait for that run "
                "to finish before sending another message."
            )
            st.text_input(
                "Message UIGPT...",
                value="",
                placeholder="Another chat is still generating a reply...",
                disabled=True,
                label_visibility="collapsed",
                key="processing_chat_input",
            )
        else:
            composer_col, action_col = st.columns([5, 1])
            with composer_col:
                st.text_input(
                    "Message UIGPT...",
                    value="",
                    placeholder="UIGPT is processing your request...",
                    disabled=True,
                    label_visibility="collapsed",
                    key="processing_chat_input",
                )
            with action_col:
                if st.button(
                    "Terminate",
                    key="btn_terminate_generation",
                    use_container_width=True,
                    type="secondary",
                ):
                    active_thread_id = st.session_state.active_thread
                    _cancel_background_run(st.session_state.active_run_id)
                    st.session_state.active_run_id = None
                    try:
                        st.session_state.thread_messages[active_thread_id] = bridge.switch_chat(active_thread_id)
                    except Exception:
                        pass
                    _set_stream_interaction_lock(False)
                    st.rerun()
    else:
        if query := st.chat_input("Message UIGPT..."):
            origin_thread_id = st.session_state.active_thread
            cached_messages = list(_get_thread_messages(origin_thread_id))
            cached_messages.append({"role": "user", "content": query})
            assistant_idx = len(cached_messages)
            cached_messages.append({"role": "assistant", "content": ""})
            st.session_state.thread_messages[origin_thread_id] = cached_messages
            st.session_state.active_run_id = _start_background_run(
                query=query,
                thread_id=origin_thread_id,
                assistant_idx=assistant_idx,
                threads=st.session_state.threads,
            )
            _set_stream_interaction_lock(True)
            st.rerun()

    if _chat_panel_refresh_interval is not None and not has_running_generation:
        st.rerun()


_render_chat_panel()
