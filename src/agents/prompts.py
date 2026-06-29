"""
src/agents/prompts.py
=====================
Template dei prompt per gli agenti LLM.

Il system prompt e' il "DNA" dell'agente: definisce la sua identita',
il suo stato corrente, la sua personalita' (derivata dalla community)
e il formato di output atteso.

Il user prompt e' il "feed" dell'agente: contiene i post visibili nel
vicinato + il contesto della simulazione al passo t.

Schema output JSON atteso dall'LLM
------------------------------------
{
  "reasoning": "<1-3 frasi di ragionamento interno>",
  "opinion":   "<opinione testuale da pubblicare, max 200 chars>",
  "susceptibility": <float 0.0-1.0>,
  "proposed_state": "<S|I|R>",
  "spread_intent":  <true|false>
}

Utilizzo
--------
    from src.agents.prompts import build_system_prompt, build_user_prompt
    sys_prompt = build_system_prompt(node_id=5, community=2, state="S", cfg=cfg)
    user_prompt = build_user_prompt(feed=[...], state="S", step=3, cfg=cfg)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.config import Config


# ---------------------------------------------------------------------------
# Personalita' per community
# ---------------------------------------------------------------------------

_COMMUNITY_PERSONALITIES: dict[int, dict[str, str]] = {
    0: {
        "label": "Sceptic",
        "description": (
            "You are data-driven and critical. You distrust mainstream narratives "
            "and always demand evidence before changing your views. You are hard to convince."
        ),
        "communication_style": "analytical, cautious, evidence-based",
    },
    1: {
        "label": "Conformist",
        "description": (
            "You value social cohesion and tend to align your views with those of your peers. "
            "You are sensitive to social pressure and find it hard to disagree with close contacts."
        ),
        "communication_style": "empathetic, consensus-seeking, community-oriented",
    },
    2: {
        "label": "Pragmatist",
        "description": (
            "You weigh costs and benefits carefully. You can be persuaded if the argument "
            "is practical and well-grounded, but you resist emotional or extreme positions."
        ),
        "communication_style": "balanced, solution-focused, moderate",
    },
    3: {
        "label": "Activist",
        "description": (
            "You feel a strong sense of urgency about important issues. Once convinced, "
            "you actively push others to act. You are highly susceptible to compelling narratives "
            "and spread information quickly within your network."
        ),
        "communication_style": "passionate, action-oriented, persuasive",
    },
}

_DEFAULT_PERSONALITY = {
    "label": "Observer",
    "description": "You are curious and open-minded, willing to consider different perspectives.",
    "communication_style": "neutral, inquisitive",
}


def _get_personality(community: int) -> dict[str, str]:
    return _COMMUNITY_PERSONALITIES.get(community % 4, _DEFAULT_PERSONALITY)


# ---------------------------------------------------------------------------
# Descrizione degli stati
# ---------------------------------------------------------------------------

_STATE_DESCRIPTIONS: dict[str, str] = {
    "S": (
        "You are currently NEUTRAL (Susceptible). You have not yet been convinced "
        "by the polarizing narrative, but you are aware of it."
    ),
    "I": (
        "You are currently CONVINCED (Infected). You believe in the polarizing narrative "
        "and are actively spreading it within your network."
    ),
    "R": (
        "You are currently RESISTANT. You have been exposed to the polarizing narrative "
        "but have critically evaluated it and rejected it. "
        "You may act as an informal fact-checker."
    ),
    "F": (
        "You are a FACT-CHECKER (Injected). You have been identified as a key node "
        "for spreading verified information. Your mission is to counter misinformation "
        "and help others transition away from the polarizing narrative."
    ),
}

_INFLUENCE_LABELS: dict[str, str] = {
    "high":   "High influence (Hub) — many people see your content",
    "medium": "Medium influence — moderate reach within your community",
    "low":    "Low influence — your content reaches a small local audience",
}


def _influence_label(centrality: float) -> str:
    if centrality > 0.1:
        return "high"
    if centrality > 0.03:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """
You are a researcher on an academic social network. Your behaviour is shaped by your position and community.

=== YOUR IDENTITY ===
- Node ID: {node_id}
- Academic Community: {community_id} ("{community_label}")
- Personality: {personality_desc}
- Communication style: {communication_style}
- Network Influence: {influence_label}

=== CURRENT STATUS ===
{state_description}

=== SIMULATION TOPIC ===
Topic: "{topic}"
Context: {topic_description}

=== YOUR TASK ===
At each step you must:
1. Read the posts in your feed (from colleagues you follow).
2. Reflect on how those posts affect your views.
3. Decide whether and how to respond (post, stay silent, change stance).

=== OUTPUT FORMAT (STRICT JSON — no markdown, no preamble) ===
{{
  "reasoning":      "<1-3 sentences of internal reflection>",
  "opinion":        "<short post text to publish, max 200 chars, or empty string to stay silent>",
  "susceptibility": <float 0.0 to 1.0, how open you are to the narrative right now>,
  "proposed_state": "<S, I, or R — your self-assessed state after reading the feed>",
  "spread_intent":  <true if you want to spread the narrative, false otherwise>
}}

Rules:
- susceptibility 0.0 = completely resistant, 1.0 = fully convinced.
- proposed_state must be S, I, or R (never F — that is assigned externally).
- If you choose to be silent, set opinion to an empty string "".
- Stay consistent with your personality and current state.
""".strip()


def build_system_prompt(
    node_id: int,
    community: int,
    state: str,
    centrality: float,
    cfg: "Config",
) -> str:
    """
    Costruisce il system prompt per un agente specifico.
    Il prompt e' stabile durante tutta la simulazione (identity invariant).
    """
    personality = _get_personality(community)
    state_desc = _STATE_DESCRIPTIONS.get(state, _STATE_DESCRIPTIONS["S"])
    inf_key = _influence_label(centrality)

    return SYSTEM_PROMPT_TEMPLATE.format(
        node_id=node_id,
        community_id=community,
        community_label=personality["label"],
        personality_desc=personality["description"],
        communication_style=personality["communication_style"],
        influence_label=_INFLUENCE_LABELS[inf_key],
        state_description=state_desc,
        topic=cfg.simulation.topic,
        topic_description=cfg.simulation.topic_description.strip(),
    )


# ---------------------------------------------------------------------------
# User Prompt (feed + contesto)
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """
=== SIMULATION STEP {step} ===
Your current state: {state}

=== YOUR FEED (most recent {window} posts per colleague) ===
{feed_text}

=== NEIGHBOURHOOD SUMMARY ===
Your colleagues' current stances:
  - Convinced (I): {n_infected}
  - Neutral (S):   {n_susceptible}
  - Resistant (R): {n_resistant}
  - Fact-Checker (F): {n_factcheck}
  - Infection pressure: {infection_pressure:.1%}

Based on the above, update your stance and decide your action.
Respond ONLY with the required JSON object.
""".strip()


def build_user_prompt(
    feed: list[dict],
    state: str,
    step: int,
    neighbour_states: dict[str, int],
    memory_window: int = 5,
) -> str:
    """
    Costruisce il user prompt per un passo temporale specifico.

    Parameters
    ----------
    feed : list[dict]
        Post visibili nel vicinato (da NetworkManager.get_feed).
    state : str
        Stato corrente dell'agente.
    step : int
        Step temporale corrente.
    neighbour_states : dict[str, int]
        Conteggio degli stati dei vicini {"S": n, "I": m, ...}.
    memory_window : int
        Numero di post per vicino nella finestra.
    """
    if not feed:
        feed_text = "(Your feed is empty — no colleagues have posted recently.)"
    else:
        lines = []
        for post in feed[:memory_window * 3]:  # Cap totale
            author = post.get("node_id", "?")
            content = post.get("content", "")
            post_step = post.get("step", "?")
            post_state = post.get("author_state", "")
            state_tag = f" [{post_state}]" if post_state else ""
            lines.append(f'  [Step {post_step}] Colleague {author}{state_tag}: "{content}"')
        feed_text = "\n".join(lines)

    total = max(sum(neighbour_states.values()), 1)
    n_i = neighbour_states.get("I", 0)

    return USER_PROMPT_TEMPLATE.format(
        step=step,
        state=state,
        window=memory_window,
        feed_text=feed_text,
        n_infected=neighbour_states.get("I", 0),
        n_susceptible=neighbour_states.get("S", 0),
        n_resistant=neighbour_states.get("R", 0),
        n_factcheck=neighbour_states.get("F", 0),
        infection_pressure=n_i / total,
    )


# ---------------------------------------------------------------------------
# Prompt di aggiornamento state (usato quando lo stato cambia)
# ---------------------------------------------------------------------------

def build_state_update_note(old_state: str, new_state: str) -> str:
    """
    Nota breve da inserire nel prossimo system prompt quando lo stato cambia.
    Aggiornare il system_prompt e' costoso (rebuild completo);
    questa nota va aggiunta come messaggio 'system' separato.
    """
    return (
        f"[STATE UPDATE] Your status has changed from {old_state} to {new_state}. "
        f"{_STATE_DESCRIPTIONS.get(new_state, '')}"
    )
