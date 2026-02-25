"""
ai_brain.py — LLM interface for market analysis.

Responsibilities:
  - Multi-provider support: OpenAI, Anthropic, Google Gemini
  - System prompt engineering with JSON-only mandate
  - Token budget guard
  - Exponential backoff retry (1s → 2s → 4s → 8s)
  - Default HOLD fallback on failure or unparseable output
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import google.generativeai as genai
import aiohttp
from aiolimiter import AsyncLimiter

from openai import AsyncOpenAI
from config import (
    LLM_PROVIDER, LLM_MAX_RETRIES, LLM_BACKOFF_BASE,
    LLM_MAX_TOKENS, LLM_TEMPERATURE,
    OPENAI_API_KEY, OPENAI_MODELS, OPENAI_BASE_URL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    MIN_CONFIDENCE_TO_TRADE,
    AI_CONSENSUS_THRESHOLD,
)
from logger import get_logger

log = get_logger("ai_brain")

# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS: int = 6000   # ~1500 tokens; safe for all models
HOLD_FALLBACK: dict = {
    "signal": "HOLD",
    "reasoning": "AI unavailable or response unparseable — defaulting to HOLD.",
    "entry_params": {"suggested_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0},
    "confidence_score": 0.0,
    "risk_assessment": "High",
}

SYSTEM_PROMPT = """You are a master Smart Money Concepts (SMC) & ICT specialist. 
Your ONLY output must be a single valid JSON object. 

CORE TRADING RULES:
1. Liquidity Hunt: Identify where retail stops are resting (Liquidity Pools).
2. Fair Value Gap (FVG): Prioritize entries that fill price imbalances.
3. Order Blocks (OB): Only trade from institutional supply/demand zones.
4. Logic First: If the price structure is messy, do NOT force a trade.

JSON Schema:
{
  "signal": "BUY" | "SELL" | "HOLD" | "EXIT",
  "reasoning": "<short SMC-based justification, max 10 words>",
  "entry_params": {"suggested_price": float, "stop_loss": float, "take_profit": float},
  "confidence_score": float (0-1),
  "risk_assessment": "High" | "Medium" | "Low"
}

Strategic Bias:
1. FOCUS on DISPLACEMENT: Look for strong moves leaving FVGs behind.
2. EXIT STRATEGY: Use "EXIT" if price hits a major liquidity draw or reverses trend.
3. STRICT JSON: NO prose, NO markdown.
"""

FEW_SHOT_EXAMPLE = """
Example of a valid response:
{
  "signal": "BUY",
  "reasoning": "Price rejected EMA-200 support with bullish engulfing candle; RSI rising from 32 indicating oversold recovery.",
  "entry_params": {
    "suggested_price": 1.08542,
    "stop_loss": 1.08300,
    "take_profit": 1.09100
  },
  "confidence_score": 0.82,
  "risk_assessment": "Medium"
}
"""

class AIBrain:
    """
    Stateful LLM interface. Provider-agnostic analysis engine.
    All calls are async-first.
    """

    def __init__(self) -> None:
        self._consecutive_failures: int = 0
        self._openai_client: Optional[AsyncOpenAI] = None
        self._session: Optional[aiohttp.ClientSession] = None
        from config import MAX_CONCURRENT_AI_CALLS, LLM_PROVIDER
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_CALLS)
        # Accurate Rate Limiting: 2 requests per 5 seconds (adjustable)
        self._limiter = AsyncLimiter(2, 5)
        log.info("AIBrain initialized", provider=LLM_PROVIDER, concurrent_lanes=MAX_CONCURRENT_AI_CALLS)

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL or "https://apis.iflow.cn/v1"
            )
        return self._openai_client

    async def _get_session(self) -> aiohttp.ClientSession:
        # Still needed for Anthropic/Gemini if we don't upgrade them yet
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        if self._openai_client:
            await self._openai_client.close()

    # ── Public API ────────────────────────────────────────────────

    async def analyze(self, market_report: str, strategy_metadata: Optional[dict] = None) -> dict:
        """
        Send market_report to LLM; return parsed JSON signal dict.
        Supports multi-model ensemble for OpenAI provider.
        strategy_metadata: {name, rules} to guide the AI's context.
        """
        # Token budget guard
        report = market_report[:MAX_CONTEXT_CHARS]
        
        context_str = ""
        if strategy_metadata:
            context_str = f"Applied Strategy: {strategy_metadata.get('name')}\nTechnical Rules: {strategy_metadata.get('rules')}\n\n"

        user_message = f"{context_str}{FEW_SHOT_EXAMPLE}\n\n{report}\n\nProvide your JSON analysis:"

        if LLM_PROVIDER == "openai" and len(OPENAI_MODELS) > 1:
            return await self._analyze_ensemble(user_message, market_report)

        # Single provider logic
        backoff_schedule = [10, 20, 60]
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            async with self._semaphore:
                try:
                    async with self._limiter:
                        t0 = time.perf_counter()
                        raw = await self._call_provider(user_message)
                        latency = round((time.perf_counter() - t0) * 1000, 1)
                        
                    parsed = self._parse_response(raw)
                    self._consecutive_failures = 0
                    log.info("AI analysis complete", symbol=market_report.split("|")[0][:10], signal=parsed.get("signal"), latency_ms=latency)
                    return parsed
                except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "449" in exc_str or "RateLimitError" in exc_str:
                        wait = backoff_schedule[min(attempt-1, len(backoff_schedule)-1)]
                        log.warning(f"Rate limit hit ({exc_str}). Applying exponential backoff: {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    
                    await self._handle_failure(attempt, exc)
        return dict(HOLD_FALLBACK)

    async def _analyze_ensemble(self, user_message: str, market_report: str) -> dict:
        """Runs multiple models in parallel and aggregates results."""
        log.info("Starting AI Ensemble analysis", models=OPENAI_MODELS)
        tasks = [self._call_openai_safe(model, user_message) for model in OPENAI_MODELS]
        results = await asyncio.gather(*tasks)
        
        valid_results = [r for r in results if r["signal"] != "HOLD" or r.get("confidence_score", 0) > 0]
        if not valid_results:
            log.warning("Ensemble failed: No valid results from any model")
            return dict(HOLD_FALLBACK)

        return self._aggregate_ensemble_results(valid_results, market_report)

    async def _call_openai_safe(self, model: str, user_message: str) -> dict:
        """Wrapper for single OpenAI model call with parsing."""
        async with self._semaphore:
            backoff_schedule = [10, 20, 60]
            for attempt in range(1, 3): # 2 attempts per model in ensemble
                try:
                    async with self._limiter:
                        raw = await self._call_openai(user_message, model_override=model)
                        parsed = self._parse_response(raw)
                        parsed["model"] = model
                    
                    return parsed
                except Exception as e:
                    exc_str = str(e)
                    if "449" in exc_str or "429" in exc_str:
                        wait = backoff_schedule[min(attempt-1, len(backoff_schedule)-1)]
                        log.warning(f"Rate limit hit for {model}. Backing off {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    log.warning(f"Model {model} failed", error=exc_str)
                    return dict(HOLD_FALLBACK)
            return dict(HOLD_FALLBACK)

    def _aggregate_ensemble_results(self, results: list[dict], market_report: str) -> dict:
        """Consensus logic: Majority rule for signal, average for confidence."""
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0, "EXIT": 0}
        confidences = []
        reasoning_list = []
        
        for r in results:
            sig = r.get("signal", "HOLD")
            votes[sig] += 1
            confidences.append(r.get("confidence_score", 0.0))
            reasoning_list.append(f"[{r.get('model', 'AI')}]: {r.get('reasoning', 'N/A')}")

        # Determine winner using Weighted Voting (Confidence * Vote)
        weighted_votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0, "EXIT": 0.0}
        for r in results:
            sig = r.get("signal", "HOLD")
            conf = r.get("confidence_score", 0.5)
            weighted_votes[sig] += conf

        final_signal = "HOLD"
        max_weight = 0.0
        
        # Tie-breaker logic: Weighted majority
        for sig, weight in weighted_votes.items():
            if sig == "EXIT": continue
            if weight > max_weight:
                max_weight = weight
                final_signal = sig
            elif weight == max_weight and weight > 0:
                # If weights are exactly equal, prioritize safety
                if sig == "HOLD" or final_signal == "HOLD":
                    final_signal = "HOLD"
                elif sig != final_signal:
                    final_signal = "HOLD" # Indecision between BUY/SELL = HOLD
        
        # Absolute vote count still matters for consensus threshold
        max_votes = votes.get(final_signal, 0)
        
        # Check if EXIT is strong
        if weighted_votes.get("EXIT", 0.0) > (sum(confidences) / 2):
            final_signal = "EXIT"
            max_votes = votes.get("EXIT", 0)

        # Apply threshold
        if final_signal not in ("HOLD", "EXIT") and max_votes < AI_CONSENSUS_THRESHOLD:
            log.info("Consensus not reached", votes=votes, threshold=AI_CONSENSUS_THRESHOLD)
            final_signal = "HOLD"

        avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
        summary_reasoning = f"Ensemble Verdict: {final_signal} ({max_votes}/{len(results)} votes). " + " | ".join(reasoning_list[:2])

        log.info("Ensemble consensus complete", signal=final_signal, votes=votes, avg_conf=avg_conf)
        
        # Return composite result
        return {
            "signal": final_signal,
            "reasoning": summary_reasoning,
            "entry_params": results[0].get("entry_params", HOLD_FALLBACK["entry_params"]) if final_signal != "HOLD" else HOLD_FALLBACK["entry_params"],
            "confidence_score": avg_conf,
            "risk_assessment": results[0].get("risk_assessment", "High"),
            "votes": votes,
            "ensemble_meta": results
        }

    async def analyze_trade_outcome(self, trade_data: dict) -> str:
        """
        Ensemble analysis of a finished trade to explain success/failure.
        trade_data: {symbol, profit, entry_price, close_price, type}
        """
        status = "SUCCESS" if trade_data['profit'] >= 0 else "FAILURE"
        prompt = f"""
        Analyze this finished trade:
        Symbol: {trade_data['symbol']}
        Type: {trade_data['type']}
        Outcome: {status} ({trade_data['profit']:.2f} USD)
        Entry: {trade_data['entry_price']}
        Exit: {trade_data['close_price']}
        
        Explain in ONE SHORT ARABIC SENTENCE why this happened based on technical logic.
        Be a professional quant analyst. 
        """
        
        log.info("Starting AI Post-Mortem", symbol=trade_data['symbol'], pnl=trade_data['profit'])
        try:
            # Use top model for high-quality post-mortem
            raw = await self._call_openai(prompt, model_override=OPENAI_MODELS[0])
            return raw.strip()
        except Exception as e:
            log.warning("AI Post-Mortem failed", error=str(e))
            return "تحليل فني سريع: تم إغلاق الصفقة وفقاً لإحصائيات السوق وتحركات السعر الحالية."

    async def _handle_failure(self, attempt: int, exc: Exception):
        wait = LLM_BACKOFF_BASE * (2 ** (attempt - 1))
        self._consecutive_failures += 1
        log.warning("AI call failed", attempt=attempt, error=str(exc), retry_in=wait)
        if attempt < LLM_MAX_RETRIES:
            await asyncio.sleep(wait)

    # ── Provider Dispatch ─────────────────────────────────────────

    async def _call_provider(self, user_message: str) -> str:
        if LLM_PROVIDER == "openai":
            return await self._call_openai(user_message)
        elif LLM_PROVIDER == "anthropic":
            return await self._call_anthropic(user_message)
        elif LLM_PROVIDER == "gemini":
            return await self._call_gemini(user_message)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'")

    # ── OpenAI ────────────────────────────────────────────────────
    async def _call_openai(self, user_message: str, model_override: Optional[str] = None) -> str:
        client = self._get_openai_client()
        model = model_override or OPENAI_MODELS[0]
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        
        if not response.choices:
            log.error("AI returned empty choices", response=str(response))
            return ""

        message = response.choices[0].message
        content = message.content or ""
        
        # Support for reasoning_content if provided by proxy (e.g. DeepSeek via One API)
        reasoning = ""
        if hasattr(message, "reasoning_content") and message.reasoning_content:
             reasoning = message.reasoning_content
             log.info("AI Reasoning extracted", length=len(reasoning))
        
        # If we have reasoning content but signal in content is missing, we might need more complex parsing
        # but the JSON mandate usually puts it in content.
        return content

    # ── Anthropic ─────────────────────────────────────────────────
    async def _call_anthropic(self, user_message: str) -> str:
        session = await self._get_session()
        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers,
        ) as resp:
            if resp.status == 429:
                raise RuntimeError("Anthropic rate limit (429)")
            resp.raise_for_status()
            data = await resp.json()
            return data["content"][0]["text"]

    # ── Google Gemini ─────────────────────────────────────────────
    async def _call_gemini(self, user_message: str) -> str:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            generation_config={
                "max_output_tokens": 2048,
                "temperature": LLM_TEMPERATURE,
                "response_mime_type": "application/json",
            }
        )
        response = await model.generate_content_async(user_message)
        if not response.text:
            log.error("Gemini returned empty text", response=str(response))
            return ""
        return response.text

    # ── JSON Parsing & Validation ──────────────────────────────────

    def _parse_response(self, raw: str) -> dict:
        """
        Extract and validate JSON from LLM response.
        Falls back to HOLD_FALLBACK if validation fails.
        """
        if not raw:
            return dict(HOLD_FALLBACK)
            
        # Strip markdown fences if any were leaked
        text = raw.strip()
        
        # Strip <think>...</think> blocks from models like DeepSeek-R1
        if "<think>" in text and "</think>" in text:
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            log.info("Stripped <think> tags from response")

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip("`").strip()

        try:
            parsed: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("JSON decode failed", error=str(exc), raw_snippet=raw[:200])
            return dict(HOLD_FALLBACK)

        # Schema validation
        required_keys = {"signal", "reasoning", "entry_params", "confidence_score", "risk_assessment"}
        if not required_keys.issubset(parsed.keys()):
            log.warning("Missing required keys", parsed_keys=list(parsed.keys()))
            return dict(HOLD_FALLBACK)

        if parsed.get("signal") not in ("BUY", "SELL", "HOLD", "EXIT"):
            log.warning("Invalid signal value", signal=parsed.get("signal"))
            return dict(HOLD_FALLBACK)

        confidence = float(parsed.get("confidence_score", 0))
        if not 0.0 <= confidence <= 1.0:
            parsed["confidence_score"] = max(0.0, min(1.0, confidence))

        # Safety: reject trade if confidence below minimum
        if parsed["signal"] != "HOLD" and confidence < MIN_CONFIDENCE_TO_TRADE:
            log.info(
                "Confidence too low — overriding to HOLD",
                confidence=confidence,
                min_required=MIN_CONFIDENCE_TO_TRADE,
            )
            parsed["signal"] = "HOLD"
            parsed["reasoning"] += f" [Overridden to HOLD: confidence {confidence} < {MIN_CONFIDENCE_TO_TRADE}]"

        # Logic Validator: THE SLAP — Scan for institutional-grade contradictions
        negative_keywords = [
            "dead volume", "invalid", "thin volume", "rejection", "unacceptable", 
            "risk too high", "stalled", "no momentum", "weakness", "neutral"
        ]
        
        reasoning_lower = parsed.get("reasoning", "").lower()
        if parsed["signal"] in ("BUY", "SELL"):
            for word in negative_keywords:
                if word in reasoning_lower:
                    log.warning(f"[LOGIC GUARD] AI HALLUCINATION BLOCKED - Signal was {parsed['signal']} but reasoning said: '{word}'. Forced to HOLD.")
                    parsed["signal"] = "HOLD"
                    parsed["reasoning"] = f"[LOGIC GUARD VETO: {word}] " + parsed["reasoning"]
                    parsed["confidence_score"] = 0.0
                    break

        return parsed
