# Trawler

**A retrospective prediction market content pipeline that almost worked.**

Trawler pulls resolved prediction markets from Polymarket, scores them for virality, and generates short-form narration scripts for TikTok and YouTube Shorts. The idea was simple: prediction markets are full of absurd, dramatic, and surprising outcomes — the kind of thing that stops a thumb mid-scroll. All you need is a pipeline to find the best ones and write the scripts.

The pipeline works. The scripts don't hold up. Here's what happened.

## The goal

Prediction markets are an untapped content goldmine. Thousands of markets resolve every week on Polymarket alone. People bet real money on whether Elon Musk would tweet a specific number of times, whether Luigi Mangione would smile in court, whether a crypto token would hit an arbitrary valuation on launch day. The outcomes are public, the odds histories tell a story, and most of it goes completely uncovered.

The plan was to build an automated pipeline that could:

1. **Ingest** resolved markets and their full price histories from Polymarket
2. **Score** each market on how entertaining it would be as content — surprise factor, narrative arc, absurdity, humor, shareability, and more
3. **Generate** compilation-style narration scripts grouped by theme, written in a punchy short-form voice
4. **Review** the scripts with a human in the loop before anything gets published

The whole thing would run as a CLI. Ingest, score, generate, review. Cherry-pick the best scripts, record them, post them. A content factory powered by the collective degeneracy of prediction market bettors.

## How it works

Trawler is a Python CLI with four pipeline stages.

**Ingestion** hits the Polymarket Gamma API to pull resolved events, then fetches price history for each market from the CLOB API. It uses a multi-bucket strategy — high volume, competitive ordering, tag-targeted, and niche low-liquidity — to get a diverse pool of markets rather than just the biggest ones. Everything gets stored in Postgres.

**Scoring** combines math-derived signals with LLM-powered judgment. Surprise score measures how unexpected the outcome was relative to the odds. Narrative arc score measures whether the price history tells an interesting story — wild swings and reversals versus a flat line drifting to its conclusion. Volume gets a sigmoid-capped score so mega-markets don't dominate. Then the LLM (Claude Haiku) rates each market on seven dimensions: absurdity, significance, shareability, humor, relatability, controversy, and WTF factor. Everything feeds into a weighted composite score. Markets below a volume floor get filtered out. Scoring commits per-batch so you can interrupt and resume.

**Generation** takes the top-scored markets, groups them by domain (Politics, Pop Culture, Sports, Tech/Business, Wildcard), deduplicates by entity and semantic similarity, and sends each group to the LLM with an elaborate system prompt. The prompt is opinionated — it demands specific opening structures, bans clichéd dollar-amount leads, distinguishes between "swing markets" with dramatic odds movement and "deadline bets" where the odds were uninstructive, and requires every segment to end on a quotable line.

**Review** renders the scripts in the terminal with Rich or exports them to markdown for human judgment.

## Where it fell short

The pipeline does what it's supposed to. Ingestion is solid, scoring produces reasonable rankings, the deduplication and grouping logic works. The problem is the last mile: the generated scripts.

**The model doesn't know what happened.**

LLMs have a knowledge cutoff. When Trawler asks Claude to write a narration segment about a resolved prediction market, it's asking the model to narrate real-world events that may have occurred after its training data ends. The model doesn't refuse — it confabulates. It fills in the gaps with plausible-sounding nonsense, and the result is a script that reads well but says things that didn't happen.

The most egregious example: a market about who would become the next Pope resolved with Pope Leo XIV winning. The model, having no knowledge of a Pope Leo XIV, hallucinated around the topic — despite the fact that it was writing a narration segment *for the very market where he won the papacy*. The data was right there in the prompt. The resolution field said his name. But the model's prior was stronger than the context, and it couldn't reconcile "this person won" with "I have no record of this person existing."

This wasn't a one-off. Any market that resolved based on events after the model's training cutoff was vulnerable. The model would subtly reframe outcomes, invent plausible-but-wrong context, or write around the topic in a way that sounded confident but was factually hollow. For retrospective content — content whose entire value proposition is "look at this real thing that happened" — that's fatal.

**You can't narrate history with a model that doesn't know the history.**

The market data in the prompt gives the model the *what* — who won, what the odds were, how much money was on the line. But the model needs the *why* and the *how* to write anything worth watching. Why was this surprising? What was the context? Who else was in contention? Without that background knowledge, the model either makes it up or writes generic filler. Both are useless for content that's supposed to feel like a friend showing you something wild on their phone.

## What would fix it

The core issue is context. The model needs access to factual information about the events these markets covered. Some options:

- **Web search at generation time.** Before writing each segment, search for the market topic and inject recent articles into the prompt. This is the most straightforward fix but adds latency, cost, and a dependency on search quality.
- **Richer market descriptions.** Polymarket's event descriptions are often thin. Enriching each market with a paragraph of factual context during ingestion — pulled from news APIs or scraped from the event pages — would give the model something real to work with.
- **A model with a more recent cutoff.** This fixes some cases but not all. Markets resolve in real time; there will always be a gap between the model's knowledge and the latest resolutions.

None of these are hard problems. The pipeline architecture supports all of them — the generation stage already formats rich context blocks for each market. The missing piece is grounding.

## Running it

```
docker compose up -d
pip install -e .
cp .env.example .env  # add your ANTHROPIC_API_KEY

trawler init
trawler ingest --limit 500
trawler score
trawler generate
trawler review --export
```

Requires Python 3.11+, Postgres (provided via Docker Compose), and an Anthropic API key.

## Stack

- Python, Typer, Rich
- Postgres (psycopg 3)
- Anthropic Claude (Haiku for scoring, Sonnet for generation)
- Polymarket Gamma + CLOB APIs
