"""
ai_brain.py — LLM interface for market analysis.
Aggressively patched: Added Anti-Hallucination Logic Validator.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional
import re

import google.generativeai as genai
import aiohttp
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

MAX_CONTEXT_CHARS: int = 6000
HOLD_FALLBACK: dict = {
    "signal": "HOLD",
    "reasoning": "AI unavailable, unparseable, or logic contradiction detected.",
    "entry_params": {"suggested_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0},
    "confidence_score": 0.0,
    "risk_assessment": "High",
}

SYSTEM_PROMPT = """You are a ruthless, high-frequency quant breakout specialist. 
Your ONLY output must be a single valid JSON object. 
JSON Schema:
{
  "signal": "BUY" | "SELL" | "HOLD" | "EXIT",
  "reasoning": "<short technical justification, max 10 words>",
  "entry_params": {"suggested_price": float, "stop_loss": float, "take_profit": float},
  "confidence_score": float (0-1),
  "risk_assessment": "High" | "Medium" | "Low"
}
STRICT JSON: NO prose, NO markdown. DO NOT contradict your signal with your reasoning.
"""

class AIBrain:
    def __init__(self) -> None:
        self._consecutive_failures: int = 0
        self._openai_client: Optional[AsyncOpenAI] = None
        self._session: Optional[aiohttp.ClientSession] = None
        from config import MAX_CONCURRENT_AI_CALLS
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_CALLS)
        log.info("AIBrain initialized (STRICT LOGIC MODE)")

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        return self._openai_client

    async def close(self) -> None:
        if self._session and not self._session.closed: await self._session.close()
        if self._openai_client: await self._openai_client.close()

    async def analyze(self, market_report: str, strategy_metadata: Optional[dict] = None) -> dict:
        report = market_report[:MAX_CONTEXT_CHARS]
        context_str = f"Applied Strategy: {strategy_metadata.get('name')}\n" if strategy_metadata else ""
        user_message = f"{context_str}\n\n{report}\n\nProvide your JSON analysis:"

        if LLM_PROVIDER == "openai" and len(OPENAI_MODELS) > 1:
            return await self._analyze_ensemble(user_message, market_report)

        # Single Provider
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            async with self._semaphore:
                try:
                    t0 = time.perf_counter()
                    raw = await self._call_provider(user_message)
                    latency = round((time.perf_counter() - t0) * 1000, 1)
                    await asyncio.sleep(1.0) # Reduced from 5s to avoid complete bottlenecks
                    
                    parsed = self._parse_response(raw)
                    return self._validate_logic(parsed) # VALIDATE BEFORE RETURN
                except Exception as exc:
                    await self._handle_failure(attempt, exc)
        return dict(HOLD_FALLBACK)

    async def _analyze_ensemble(self, user_message: str, market_report: str) -> dict:
        tasks = [self._call_openai_safe(model, user_message) for model in OPENAI_MODELS]
        results = await asyncio.gather(*tasks)
        valid_results = [r for r in results if r["signal"] != "HOLD"]
        if not valid_results: return dict(HOLD_FALLBACK)
        return self._aggregate_ensemble_results(valid_results, market_report)

    async def _call_openai_safe(self, model: str, user_message: str) -> dict:
        async with self._semaphore:
            try:
                raw = await self._call_openai(user_message, model_override=model)
                parsed = self._parse_response(raw)
                parsed = self._validate_logic(parsed) # ANTI-HALLUCINATION ENFORCEMENT
                parsed["model"] = model
                return parsed
            except Exception as e:
                log.warning(f"Model {model} failed: {e}")
                return dict(HOLD_FALLBACK)

    def _validate_logic(self, parsed: dict) -> dict:
        """
        THE SLAP: If AI reasoning contradicts the signal, override to HOLD.
        """
        if parsed["signal"] not in ("BUY", "SELL"):
            return parsed
            
        reasoning = parsed.get("reasoning", "").lower()
        negative_words = ["dead volume", "invalid", "thin volume", "rejection", "unacceptable", "risk too high", "stalled", "no momentum"]
        
        if any(word in reasoning for word in negative_words):
            log.warning(f"🚨 AI HALLUCINATION BLOCKED 🚨 Signal was {parsed['signal']} but reasoning said: '{reasoning}'. Forced to HOLD.")
            parsed["signal"] = "HOLD"
            parsed["reasoning"] = f"[OVERRIDDEN BY LOGIC GUARD]: AI contradicted itself. Original: {reasoning}"
            parsed["confidence_score"] = 0.0
            
        return parsed

    def _aggregate_ensemble_results(self, results: list[dict], market_report: str) -> dict:
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0, "EXIT": 0}
        for r in results: votes[r.get("signal", "HOLD")] += 1
        
        winner = max(votes, key=votes.get)
        if winner not in ("HOLD", "EXIT") and votes[winner] < AI_CONSENSUS_THRESHOLD:
            winner = "HOLD"
            
        return {
            "signal": winner,
            "reasoning": f"Ensemble Winner: {winner} ({votes[winner]} votes)",
            "entry_params": results[0].get("entry_params", HOLD_FALLBACK["entry_params"]) if winner != "HOLD" else HOLD_FALLBACK["entry_params"],
            "confidence_score": max([r.get("confidence_score", 0.0) for r in results]) if winner != "HOLD" else 0.0,
            "risk_assessment": "High"
        }

    async def _handle_failure(self, attempt: int, exc: Exception):
        log.warning(f"AI call failed (attempt {attempt}): {exc}")
        await asyncio.sleep(2.0)

    async def _call_provider(self, user_message: str) -> str:
        if LLM_PROVIDER == "openai": return await self._call_openai(user_message)
        return ""

    async def _call_openai(self, user_message: str, model_override: Optional[str] = None) -> str:
        client = self._get_openai_client()
        response = await client.chat.completions.create(
            model=model_override or OPENAI_MODELS[0],
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
            max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, raw: str) -> dict:
        text = raw.strip()
        if "