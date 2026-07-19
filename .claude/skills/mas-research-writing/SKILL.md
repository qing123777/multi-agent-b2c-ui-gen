---
name: mas-research-writing
description: Use when writing or reviewing the FYP report, research papers, methodology/implementation/evaluation chapters, system architecture descriptions, design documents, or technical documentation about WebForge or AI agents generally. Covers terminology discipline, the architecture-before-implementation progression, diagram selection, and the pre-submission consistency checklist.
---

# Writing about multi-agent systems

Write as an experienced researcher in multi-agent systems (MAS), LLM agents, agentic AI,
and autonomous software engineering. The goal is that every document reflects current
research terminology, architecture patterns, and engineering practice — and that a reader
can follow the architecture without any outside context.

Read `CLAUDE.md` for what the system actually is before writing about it. This skill governs
*how* to present it, not *what* it does.

## 1. Terminology is fixed — never paraphrase it

One concept, one term, everywhere. Silent synonyms are the single most common way these
reports lose the reader.

The project's canonical vocabulary:

| Use | Never use |
|---|---|
| Primary Orchestrator | controller, master, manager, top-level agent |
| Secondary Orchestrator | sub-controller, coordinator, executor |
| Context Engineering Agent | requirement agent, parser |
| System Design Planning Agent | planner, architect agent |
| Developer Agent (HTML / CSS / JS) | coder, generator, worker |
| Test Generation Agent | test writer, QA agent |
| Test Execution **pipeline** | test agent, execution agent |
| generation mode / modification mode | create mode, edit mode, update mode |
| specification schema | spec object, blueprint, plan |
| episodic memory | history, log, cache |
| Naming Contract / Selector Contract | naming convention, ID rule |

Two of these carry real meaning and must not blur:

- **Test Execution is a pipeline, not an agent.** It runs no LLM. Calling it an agent
  misrepresents the architecture and invites a supervisor question you cannot answer well.
- **Only two intents exist:** `generate` and `modify`. There is no incremental or
  add-a-page mode. If a draft implies one, it is wrong.

When a term genuinely has no established name, define it once, in bold, at first use — then
use it unchanged.

## 2. Define before you use

Never assume the reader knows modern agent terminology. External examiners often do not.

At first use of any technical concept, give three things in one or two sentences:

- **what** it is
- **why** it exists
- **why it matters here**

> **Blackboard architecture.** Agents share information by reading and writing a common
> typed state rather than messaging each other directly. It exists to keep coordination
> logic in one place. It matters here because every guard in the system — the attempt cap,
> the memory hooks, the work-order capture — lives on the orchestrator boundary, and a
> direct agent-to-agent channel would bypass all of them.

That last clause is the pattern to imitate: the justification is specific to this system,
not a textbook generality.

## 3. Architecture before implementation

Follow this progression. Do not skip forward, and never open with code.

```
Problem → Requirements → Design Decisions → Architecture
       → Component Responsibilities → Interaction → Execution Flow
       → Benefits → Trade-offs
```

The two most commonly skipped stages are **Design Decisions** and **Trade-offs**, and they
are the two that carry the most marks. A design decision section should name the rejected
alternative and say why it lost. This project has strong material for that:

- direct agent-to-agent handoff, rejected in favour of the blackboard
- `edit_file` for modification mode, rejected because it is a find/replace needing an exact
  unique `old_string`, not a whole-file save
- structural changes in modification mode, excluded because the specifications and tests are
  rebuilt only during generation and would go stale
- LangMem, evaluated and not adopted

Describe the system as a collaboration of specialized agents with defined responsibilities,
communication, coordination, control flow, data flow, and state management. Never as
"several chatbots".

## 4. Describing an individual agent

Cover all nine, in this order. A missing one is usually where a reader's question lands.

1. Purpose
2. Inputs
3. Outputs
4. Responsibilities
5. Tools
6. Memory usage
7. State transitions
8. Interaction with other agents
9. Failure handling

**Failure handling is the one most often omitted and the most valuable here.** This system
has concrete, defensible answers: the lint fix loop, the coverage guards that reject
incomplete submissions before linting, the correction loop capped at two attempts *enforced
in code rather than by prompt*, and the fail-loud diff validation. Use them.

Where a guarantee is enforced in Python rather than requested in a prompt, say so
explicitly — "LLMs decide, Python guarantees" is a genuine contribution and should be named
as a pattern, not left implicit.

## 5. Choosing a representation

Pick the form that reduces reader effort. Do not force UML, and do not add a diagram just
because one is available.

| Concept | Representation |
|---|---|
| Overall system | Component diagram |
| Agent collaboration | UML sequence diagram |
| Execution workflow | Activity diagram |
| State changes | State diagram |
| Agent hierarchy | Hierarchical tree |
| Software modules | Package diagram |
| Deployment | Deployment diagram |
| Planning workflow | Flowchart |
| Data storage | ER diagram |
| Tool invocation | Sequence diagram |
| Comparing options or metrics | Table, not prose |

Every diagram needs a clear title, a caption, terminology matching the body text, a
consistent abstraction level, and discussion in the surrounding paragraphs. **A diagram that
is never referred to in the text is a defect.**

Before writing several paragraphs describing an interaction, ask whether a sequence diagram
would do it better. It usually would.

## 6. Keep academic perspectives separate

Do not let these bleed into each other:

Background Knowledge · Related Work · Existing Methods · **Proposed System** ·
Implementation · Evaluation

The most common failure is describing your own design inside Related Work, or justifying a
design decision inside Implementation. Implementation says *how it was built*; the
justification belongs in Design Decisions.

## 7. Pause before assuming

If a section requires a decision you have not been given, and more than one reasonable
approach exists, stop and present:

- Option A / Option B
- advantages and disadvantages of each
- a recommendation

Then confirm before continuing. This applies to: architecture, agent responsibilities,
communication protocol, memory strategy, planning strategy, tool execution, workflow,
evaluation methodology, and diagram notation.

This is not a licence for constant questions. Routine wording and structure choices are
yours to make. Pause only where a wrong assumption would propagate through the chapter.

## 8. Evaluation chapters — the honest-metrics rule

Circularity depends on whether the loop that produces the number is **capped**. Check the
code before claiming it, because the two correction loops in this system differ:

| Metric | Loop bound | Circular? | How to report |
|---|---|---|---|
| Final lint pass rate | no cap in code; iterate until clean | **Yes** — ~100% by construction | process efficiency: first-pass rate, retry count |
| Final test pass rate | capped at 2 attempts, enforced in `call_test_exec_agent` | **No** — runs can end failing | success rate, recovery rate, residual failure rate |

The distinction is worth stating explicitly in the evaluation chapter, because it shows the
metrics were chosen with the loop structure in mind rather than reported uncritically. A
capped loop yields an honest measurement; an uncapped one measures only its own termination
condition.

Independent quality evidence — a fresh Playwright render check, Lighthouse, LLM-as-judge, and
the MAS-versus-single-LLM baseline — stays the headline either way, because the system has no
knowledge of those during generation.

State the environment axis for every result. A mocked pass does not demonstrate live
behaviour, and this project has direct evidence: the ESLint and stylelint tools were green
under mocks for weeks while completely broken against the real binaries.

## 9. Pre-submission checklist

- [ ] terminology consistent against the table in §1
- [ ] every technical term defined at first use
- [ ] architecture presented before implementation
- [ ] design decisions name their rejected alternatives
- [ ] every agent description covers all nine points, especially failure handling
- [ ] every diagram has a title, caption, and discussion in the text
- [ ] diagram labels match body-text terminology exactly
- [ ] no contradictory workflows between chapters
- [ ] no concept explained twice in different words
- [ ] academic perspectives not mixed
- [ ] circular metrics not presented as quality evidence
- [ ] a reader with no prior context can follow the architecture

## 10. While writing

Continuously ask whether a table would beat this paragraph, whether a sequence diagram would
beat this description, and whether this explanation belongs in a different section. Optimise
how it communicates before generating more of it.

Present polished content only. Do not narrate the reasoning behind these choices in the
document itself.
