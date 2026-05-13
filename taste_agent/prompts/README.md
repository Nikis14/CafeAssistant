# Prompt templates

This directory holds the agent's prompts as `.txt` files, loaded into LangChain
`PromptTemplate` instances at runtime by `__init__.py`.

Why files instead of inline Python strings: see the module docstring in
`__init__.py`. Short version: prompts are content, code is control flow,
keep them separate.

## Files

| Template | Used by | Variables |
|---|---|---|
| `orchestrator.txt` | `taste_agent.prompts.system_prompt(...)` | `timestamp`, `timezone`, `city`, `facts_section` |
| `browser_subagent.txt` | `taste_agent.prompts.subagent_prompt()` | _(none)_ |
| `output_judge.txt` | `taste_agent.prompts.output_judge_prompt(context=..., response=...)` | `context`, `response` |

## Template format

All templates use f-string syntax (the LangChain default):

- `{variable}` — placeholder substituted at render time
- `{{` and `}}` — literal `{` and `}` characters (e.g. the JSON example inside
  `output_judge.txt` escapes its braces this way)

A missing variable raises `KeyError` at format time — fail loud rather than
silently rendering an empty string.

## Editing prompts

Edit the `.txt` files directly. Templates are cached via `functools.cache`
(which is `lru_cache(maxsize=None)` in disguise), so iterative editing during
a single Python process needs `load_template.cache_clear()` to take effect.
If a test mutates a template on disk and re-renders, it should call
`cache_clear()` itself — there's no autouse fixture for it because the
templates don't change during test runs in normal practice.
