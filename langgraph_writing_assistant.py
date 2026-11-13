"""
LangGraph Writing Assistant (Part 3 of 3: LangGraph)
   Drafts professional emails and corrects English grammar.
   Runs 100% on your computer with a 4-bit quantized model. No external API.

Quick by default. Add --think to see the model's THINKING live (grey);
the RESULT is always printed in one piece at the end.

Menu (same as the LangChain script):
    1) Draft email            2) Revise last email
    3) Correct English        4) Correct many lines
    5) Email agent (finds real dates)     h) Show saved steps of the last run

How the graph works:
    EMAIL:   write -> (tools, if it needs a date) -> review -> fix (once, if needed)
             -> ask you: Enter = approve, or type a change -> write again
    REVISE:  add your change to the saved email conversation -> write -> ...
    CORRECT: split into sentences (or lines) -> correct each one in parallel -> combine
    AGENT:   a small separate graph: model <-> tools, until it has an answer

LangGraph features used here (search for "FEATURE"):
    1. State             - one shared dict that every step reads and updates
    2. Nodes + edges     - the steps and the arrows between them
    3. Conditional edges - choose the next step (email, revise or correct)
    4. Loop              - write -> review -> fix -> write
    5. Tools             - ToolNode + tools_condition (look up real dates)
    6. Human-in-the-loop - interrupt() pauses so you can approve or change
    7. Memory            - InMemorySaver + thread_id (pause/resume, revise last email)
    8. Parallel (Send)   - each sentence or line is corrected in its own branch
    9. Retry             - RetryPolicy tries a failed step again
   10. Streaming         - see each step finish as it happens
   11. Agent by hand     - MessagesState + model node + ToolNode in a loop
   12. Cache             - correcting the same sentence again is instant
   13. Time travel       - list every saved step of the last run (menu: h)

Run:
    python langgraph_writing_assistant.py                          # menu
    python langgraph_writing_assistant.py email "ask boss for next Friday off"
    python langgraph_writing_assistant.py correct "he go to school. me and him is late"
    python langgraph_writing_assistant.py agent "ask boss for next Friday off"
    python langgraph_writing_assistant.py --tone friendly email "invite team to lunch"
    python langgraph_writing_assistant.py --yes email "..."        # don't ask, approve
    python langgraph_writing_assistant.py --think email "..."     # show thinking (slower)
    python langgraph_writing_assistant.py --show-graph             # print the diagram
"""

import os
import subprocess
import sys

# If started with another Python, re-run with the .venv next to this file
# (or one folder up), so plain "python this_file.py" always works.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BIN = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
for _VENV in (os.path.join(_HERE, ".venv"), os.path.join(os.path.dirname(_HERE), ".venv")):
    _VENV_PY = os.path.join(_VENV, *_BIN)
    if __name__ == "__main__" and os.path.exists(_VENV_PY):
        if os.path.normcase(sys.prefix) != os.path.normcase(_VENV):
            sys.exit(subprocess.call([_VENV_PY, *sys.argv]))
        break

import argparse
import operator
import re
import time
import uuid
from datetime import date, timedelta
from typing import Annotated, TypedDict

import ollama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import CachePolicy, Command, RetryPolicy, Send, interrupt
from pydantic import BaseModel

# All 4-bit quantized (Q4_K_M). The first one you have installed is used.
MODELS = ["qwen3.5:4b", "llama3.2:3b", "qwen2.5:3b", "qwen3.5:latest"]
MAX_FIXES = 1  # how many times the reviewer may send the email back

os.system("")  # enable colours in the Windows console
GREY, BOLD, RESET = "\033[90m", "\033[1m", "\033[0m"

EMAIL_PROMPT = (
    "You write short, polite emails in a {tone} tone. Today is {today}. "
    "If the request names a weekday (like 'Friday'), FIRST call next_weekday with that day and put the returned date in the email. "
    "Output only a 'Subject:' line, then the email with greeting and sign-off. "
    "Keep the body to 2-4 short sentences. "
    "Use ONLY the facts the user gives. Do not add promises, reasons or details "
    "(no lines like 'I will finish my work first' or 'I will be available by email'). "
    "Use [placeholders] for unknown names. When asked for a change, rewrite the whole email."
)


def email_prompt(tone: str = "professional") -> str:
    return EMAIL_PROMPT.format(tone=tone, today=date.today().strftime("%A %d %B %Y"))

REVIEW_PROMPT = (
    "Check the email against the user's instructions. Problems: facts or promises the "
    "user did not give, missing information, grammar mistakes, a body longer than 4 sentences. "
    "Set ok=true if there are no real problems, else ok=false and describe the "
    "problems in ONE short sentence."
)
CORRECT_PROMPT = (
    "You are a careful English proofreader. Fix grammar, spelling, tense and word choice "
    "in this sentence. Keep the meaning. List every change (old, new, reason in 2-5 words); "
    "do not list words that stay the same. "
    "Examples: 'She goed home.' -> 'She went home.' (goed -> went). "
    "'Me and him was late.' -> 'He and I were late.' (Me and him -> He and I, was -> were)."
)


# Show the model's thinking live (a LangChain callback, works inside LangGraph too)
class ShowThinking(BaseCallbackHandler):
    thinking = False

    def on_llm_new_token(self, token, chunk=None, **kwargs):
        thought = chunk.message.additional_kwargs.get("reasoning_content") if chunk else None
        if thought:
            if not self.thinking:
                print(f"{GREY}----- thinking -----")
                self.thinking = True
            print(thought, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        if self.thinking:
            print(f"\n----- thinking done -----{RESET}")
            self.thinking = False


# FEATURE 5: Tools the email writer may call
@tool
def get_today() -> str:
    """Return today's weekday and date."""
    return date.today().strftime("%A %d %B %Y")


@tool
def next_weekday(day: str) -> str:
    """Return the date of the next given weekday, e.g. 'friday'."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if day.lower() not in days:
        return f"Unknown weekday: {day}"
    ahead = (days.index(day.lower()) - date.today().weekday() - 1) % 7 + 1
    return (date.today() + timedelta(days=ahead)).strftime("%A %d %B %Y")


TOOLS = [get_today, next_weekday]


# Answers that must have a fixed shape
class Review(BaseModel):
    ok: bool
    problems: str


class Change(BaseModel):
    old: str
    new: str
    reason: str


class Correction(BaseModel):
    corrected: str
    changes: list[Change]


# FEATURE 1: State. "Annotated" fields are added to, not replaced.
class State(TypedDict, total=False):
    task: str                                   # "email", "revise", "correct", "correct_many"
    text: str                                   # what you typed
    tone: str                                   # e.g. professional, friendly
    messages: Annotated[list, add_messages]     # email conversation
    draft: str
    problems: str
    fixes: int
    sentences: list[str]
    corrected: Annotated[list, operator.add]    # results from parallel branches
    result: str


def make_llm(model: str, think: bool, max_words: int = 600, **extra) -> ChatOllama:
    """The local model. Small context + short answers = fast on a CPU."""
    return ChatOllama(model=model, reasoning=think, temperature=0.3, keep_alive="30m",
                      num_ctx=4096 if think else 2048,
                      num_predict=2500 if think else max_words, **extra)


def build_graph(model: str, think: bool):
    writer = make_llm(model, think).bind_tools(TOOLS)
    reviewer = make_llm(model, think, max_words=400, format=Review.model_json_schema())
    corrector = make_llm(model, think, max_words=400, format=Correction.model_json_schema())

    # ----- FEATURE 2: nodes (each step is a plain function) -----
    def start_email(state):
        prompt = email_prompt(state.get("tone") or "professional")
        return {"messages": [SystemMessage(prompt), HumanMessage(state["text"])], "fixes": 0}

    def revise(state):
        # FEATURE 7: memory - this thread still has the whole email conversation saved
        return {"messages": [HumanMessage(f"Rewrite the whole email with this change: {state['text']}")],
                "fixes": 0, "result": ""}

    def write(state):
        reply = writer.invoke(state["messages"])
        return {"messages": [reply], "draft": reply.content.strip()}

    def review(state):
        asked = "\n".join(m.content for m in state["messages"] if isinstance(m, HumanMessage))
        reply = reviewer.invoke([SystemMessage(REVIEW_PROMPT),
                                 HumanMessage(f"Instructions:\n{asked}\n\nEmail:\n{state['draft']}")])
        try:
            r = Review.model_validate_json(reply.content)
        except ValueError:                              # unreadable review: skip it
            return {"problems": ""}
        return {"problems": "" if r.ok else r.problems}

    def fix(state):
        return {"messages": [HumanMessage(f"Fix these problems: {state['problems']}")],
                "fixes": state["fixes"] + 1}

    def ask_human(state):
        # FEATURE 6: pause here; the value you type comes back as "answer"
        answer = interrupt(state["draft"])
        if not answer.strip():
            return {"result": state["draft"]}           # approved
        return {"messages": [HumanMessage(f"Rewrite the email with this change: {answer}")]}

    def split(state):
        if state["task"] == "correct_many":
            parts = state["text"].splitlines()                       # one per line
        else:
            parts = re.split(r"(?<=[.!?])\s+", state["text"].strip())  # one per sentence
        return {"sentences": [s.strip() for s in parts if s.strip()]}

    def correct_one(job):                               # gets {"i": ..., "sentence": ...}
        reply = corrector.invoke([SystemMessage(CORRECT_PROMPT), HumanMessage(job["sentence"])])
        try:
            c = Correction.model_validate_json(reply.content)
        except ValueError:                              # unreadable answer: keep the sentence
            c = Correction(corrected=job["sentence"], changes=[])
        return {"corrected": [{"i": job["i"], **c.model_dump()}]}   # save plain dicts

    def combine(state):
        parts = sorted(state["corrected"], key=lambda c: c["i"])
        if state["task"] == "correct_many":                         # one result per line
            return {"result": "\n".join(f"{state['sentences'][c['i']]}\n  -> {c['corrected']}"
                                        for c in parts)}
        changes =[f"- {x['old']} -> {x['new']} ({x['reason']})"
                   for c in parts for x in c["changes"] if x["old"] != x["new"]]
        return {"result": "Corrected: " + " ".join(c["corrected"] for c in parts)
                          + "\nChanges:\n" + ("\n".join(changes) or "- none")}

    # ----- FEATURE 3: functions that choose the next step -----
    def pick_task(state):
        return {"email": "start_email", "revise": "revise",
                "correct": "split", "correct_many": "split"}[state["task"]]

    def fix_or_ask(state):                              # FEATURE 4: loop back to "fix"
        return "fix" if state["problems"] and state["fixes"] < MAX_FIXES else "ask_human"

    def done_or_rewrite(state):
        return END if state.get("result") else "write"

    def one_branch_per_sentence(state):                 # FEATURE 8: Send = parallel
        return [Send("correct_one", {"i": i, "sentence": s}) for i, s in enumerate(state["sentences"])]

    # ----- build the graph -----
    retry = RetryPolicy(max_attempts=2)                 # FEATURE 9
    g = StateGraph(State)
    g.add_node("start_email", start_email)
    g.add_node("revise", revise)
    g.add_node("write", write, retry_policy=retry)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("review", review, retry_policy=retry)
    g.add_node("fix", fix)
    g.add_node("ask_human", ask_human)
    g.add_node("split", split)
    g.add_node("correct_one", correct_one, retry_policy=retry,
               cache_policy=CachePolicy(ttl=3600))     # FEATURE 12: same sentence = instant
    g.add_node("combine", combine)

    g.add_conditional_edges(START, pick_task, ["start_email", "revise", "split"])
    g.add_edge("start_email", "write")
    g.add_edge("revise", "write")
    g.add_conditional_edges("write", tools_condition, {"tools": "tools", END: "review"})
    g.add_edge("tools", "write")
    g.add_conditional_edges("review", fix_or_ask, ["fix", "ask_human"])
    g.add_edge("fix", "write")
    g.add_conditional_edges("ask_human", done_or_rewrite, ["write", END])
    g.add_conditional_edges("split", one_branch_per_sentence, ["correct_one"])
    g.add_edge("correct_one", "combine")
    g.add_edge("combine", END)

    # FEATURE 7: checkpointer = memory between steps and runs
    # FEATURE 12: cache = a node with the same input returns its saved answer instantly
    return g.compile(checkpointer=InMemorySaver(), cache=InMemoryCache())


def build_agent(model: str, think: bool):
    """FEATURE 11: an agent built by hand - the model and the tools take turns
    until the model answers without asking for a tool."""
    llm = make_llm(model, think).bind_tools(TOOLS)

    def agent(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    g = StateGraph(MessagesState)          # ready-made state: just a list of messages
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)   # tool call? -> tools, else -> END
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=InMemorySaver())


def run(graph, request, thread_id: str | None = None, auto_approve: bool = False) -> str:
    """Run the graph, show each step, then print the result in one piece."""
    # FEATURE 7: the thread_id is the "conversation id"; reuse it to continue
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id},
              "callbacks": [ShowThinking()], "max_concurrency": 1}
    start = time.time()

    while True:
        # FEATURE 10: streaming - print each step as soon as it finishes
        for update in graph.stream(request, config, stream_mode="updates"):
            for step, data in update.items():
                if step == "__metadata__":                  # FEATURE 12: cache hit
                    print(f"{GREY}[cache: saved answer reused, no model call]{RESET}")
                if step.startswith("__"):
                    continue
                note = ""
                if data and data.get("problems"):
                    note = f" -> problems: {data['problems']}"
                elif step == "tools":
                    note = " -> " + "; ".join(m.content for m in data["messages"])
                print(f"{GREY}[step done: {step}{note}]{RESET}")

        state = graph.get_state(config)
        if not state.next:                              # nothing left to run = finished
            break
        print(f"\n{BOLD}===== DRAFT ====={RESET}\n{state.values['draft']}")
        answer = "" if auto_approve else input("\nPress Enter to approve, or type a change:\n> ")
        request = Command(resume=answer)                # FEATURE 6: continue from the pause

    values = graph.get_state(config).values
    result = values.get("result") or values["messages"][-1].content
    print(f"\n{BOLD}===== RESULT ====={RESET}\n{result.strip()}")
    print(f"{GREY}({time.time() - start:.0f}s){RESET}")
    return thread_id


def show_history(graph, thread_id: str | None) -> None:
    """FEATURE 13: time travel - every step of a run is saved and can be listed."""
    if not thread_id:
        print("Run something first.")
        return
    config = {"configurable": {"thread_id": thread_id}}
    for snap in reversed(list(graph.get_state_history(config))):
        print(f"step {snap.metadata.get('step'):>2} | next: {', '.join(snap.next) or 'done'}")


def pick_model() -> str:
    installed = [m.model for m in ollama.list().models]
    return next((m for m in MODELS if m in installed), MODELS[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", choices=["email", "correct", "agent"])
    p.add_argument("text", nargs="?")
    p.add_argument("-m", "--model", help="Ollama model name")
    p.add_argument("--tone", default="professional")
    p.add_argument("--think", action="store_true", help="show the model's thinking (slower)")
    p.add_argument("--yes", action="store_true", help="approve the email without asking")
    p.add_argument("--show-graph", action="store_true", help="print the graph (Mermaid)")
    args = p.parse_args()

    model = args.model or pick_model()
    if args.show_graph:
        print(build_graph(model, False).get_graph().draw_mermaid())  # paste into mermaid.live
        return
    think = "thinking" in (ollama.show(model).capabilities or []) and args.think
    print(f"[langgraph | model: {model} | thinking: {'on' if think else 'off'}]")
    ollama.generate(model=model, prompt="", keep_alive="30m")  # load into RAM now
    graph = build_graph(model, think)
    agent = build_agent(model, think)

    def agent_request(text):
        return {"messages": [SystemMessage(email_prompt(args.tone)), HumanMessage(text)]}

    if args.task and args.text:
        if args.task == "agent":
            run(agent, agent_request(args.text))
        else:
            run(graph, {"task": args.task, "text": args.text, "tone": args.tone}, auto_approve=args.yes)
        return

    email_thread = last_thread = None
    while True:
        print("\n1) Draft email          2) Revise last email\n"
              "3) Correct English      4) Correct many lines\n"
              "5) Email agent (finds real dates)\n"
              "h) Show saved steps of the last run      q) Quit")
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        if choice == "1":
            text = input("What should the email say?\n> ")
            tone = input("Tone (Enter = professional): ").strip() or "professional"
            email_thread = last_thread = run(graph, {"task": "email", "text": text, "tone": tone},
                                             auto_approve=args.yes)
        elif choice == "2":
            if not email_thread:
                print("Draft an email first (option 1).")
                continue
            change = input("What should change? (e.g. 'make it shorter')\n> ")
            # same thread_id = the graph remembers the email conversation
            last_thread = run(graph, {"task": "revise", "text": change},
                              thread_id=email_thread, auto_approve=args.yes)
        elif choice == "3":
            last_thread = run(graph, {"task": "correct", "text": input("Text to correct:\n> ")})
        elif choice == "4":
            print("One sentence per line. Empty line = start.")
            lines = []
            while line := input("> ").strip():
                lines.append(line)
            last_thread = run(graph, {"task": "correct_many", "text": "\n".join(lines)})
        elif choice == "5":
            run(agent, agent_request(input("What should the email say?\n> ")))
        elif choice == "h":
            show_history(graph, last_thread)
        else:
            print("Please type 1-5, h or q.")


if __name__ == "__main__":
    main()
