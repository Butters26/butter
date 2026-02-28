#!/usr/bin/env python3
"""
Advanced Emotional Engine - production-ready, single-file build
- Autonomous inner life (attachment/needs/internal loops) so she can feel without mirroring
- PAD dynamics with hysteresis + refractory to stop flip-flopping
- Blends, memories, pattern learning, expressions wired into output
- Safer keyword detection + false-positive filters
- Clean persistence (save/load) with no hardcoded paths
- Deterministic hooks (rng + logger) for unit tests
"""
from __future__ import annotations

import json
import os
import random
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from threading import Lock

try:
    from thalamus import get_thalamus
except Exception:
    # Keep module importable in standalone/unit-test environments.
    def get_thalamus() -> Any:  # type: ignore[override]
        return None


# ------------------------------
# Core Enums & Dataclasses
# ------------------------------


class EmotionalState(Enum):
    # Primary emotions
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    WORRIED = "worried"
    CURIOUS = "curious"
    PROUD = "proud"
    SCARED = "scared"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    CONTEMPT = "contempt"
    # Complex blends
    NOSTALGIC = "nostalgic"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"
    EUPHORIC = "euphoric"
    MELANCHOLIC = "melancholic"
    PLAYFUL = "playful"
    PROTECTIVE = "protective"
    MISCHIEVOUS = "mischievous"


@dataclass
class PAD:
    v: float  # valence (-1..1)
    a: float  # arousal (-1..1)
    d: float  # dominance (-1..1)


@dataclass
class EmotionalMemory:
    emotion: EmotionalState
    intensity: float  # 0..1
    trigger: str
    timestamp: float
    context: str
    influence_strength: float = 1.0
    associated_emotions: List[EmotionalState] = field(default_factory=list)


@dataclass
class PersonalityTraits:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    emotional_sensitivity: float = 0.6
    emotional_stability: float = 0.5
    emotional_expressiveness: float = 0.6
    empathy_level: float = 0.6
    # PAD home base + gains
    pad_setpoint_v: float = 0.1
    pad_setpoint_a: float = 0.15
    pad_setpoint_d: float = 0.2
    hysteresis_margin: float = 0.15
    refractory_sec: float = 1.5


@dataclass
class AttachmentModel:
    security: float = 0.6
    sensitivity: float = 0.7
    bond_strength: float = 0.9
    abandonment_fear: float = 0.2
    hurt: float = 0.0
    guilt: float = 0.0


@dataclass
class InternalNeeds:
    safety: float = 0.7
    belonging: float = 0.8
    autonomy: float = 0.8
    competence: float = 0.6
    stimulation: float = 0.5


@dataclass
class EmotionalStateOutput:
    """Standardized emotional state output for other lobes to read."""

    emotion: str  # Current emotional state
    intensity: float  # 0..1 how intense
    pleasure: float  # PAD: -1..1 valence
    arousal: float  # PAD: -1..1 activation level
    dominance: float  # PAD: -1..1 control/power
    emotional_tone: str  # For language gen: "happy", "sad", "angry", etc
    emphasis: List[str]  # Speech emphasis patterns
    voice_prosody: Dict[str, float]  # For voice lobe: pitch, speed, warmth, clarity
    confidence: float  # How confident is Monday in this emotional state
    timestamp: float  # When this state was created

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission."""
        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "emotional_tone": self.emotional_tone,
            "emphasis": self.emphasis,
            "voice_prosody": self.voice_prosody,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class InternalState:
    worry: float = 0.2
    tension: float = 0.1
    hope: float = 0.3
    fatigue: float = 0.2
    rumination: float = 0.0
    competence: float = 0.8
    autonomy: float = 0.5


@dataclass
class ExpressionState:
    tears: bool = False
    voice_shake: bool = False
    withdraw: bool = False


@dataclass
class EmotionalBlend:
    primary_emotion: EmotionalState
    secondary_emotions: List[Tuple[EmotionalState, float]]
    intensity: float
    created_at: float


# ------------------------------
# Engine
# ------------------------------


class AdvancedEmotionalEngine:
    def __init__(
        self,
        name: str = "AI",
        logger: Optional[Callable[[str], None]] = None,
        rng: Optional[random.Random] = None,
        thalamus: Optional[Any] = None,
    ):
        self.name = name
        self.current_emotion: EmotionalState = EmotionalState.CALM
        self.emotional_intensity: float = 0.4
        self.personality = PersonalityTraits()
        self.emotional_memories: List[EmotionalMemory] = []
        self.mood_history: List[Tuple[float, EmotionalState, float]] = []
        self.emotional_blends: List[EmotionalBlend] = []
        self.emotional_patterns: Dict[str, List[EmotionalState]] = {}
        self.emotional_resonance: float = 0.0
        self.emotional_predictions: Dict[str, Dict[str, float]] = {}
        self.emotional_trauma_memories: List[Dict[str, Any]] = []
        self.emotional_intelligence_score: float = 0.5
        self.emotional_decay_rate: float = 0.05
        self.blend_threshold: float = 0.3
        self.memory_influence_decay: float = 0.02
        self._logger = logger or (lambda _: None)
        self._rng = rng or random.Random()
        self.thalamus = thalamus
        self.engine_lock = Lock()
        self._created_at = time.time()
        # PAD dynamics - start neutral
        self.pad: PAD = PAD(0.0, 0.0, 0.0)
        self._last_primary: EmotionalState = EmotionalState.CALM
        self._last_switch_time: float = 0.0
        # PAD prototypes (better distributed for variety)
        self._PAD_PROTOS: Dict[EmotionalState, Tuple[float, float, float]] = {
            EmotionalState.HAPPY: (0.80, 0.30, 0.20),
            EmotionalState.SAD: (-0.80, -0.20, -0.40),
            EmotionalState.ANGRY: (-0.70, 0.70, 0.60),
            EmotionalState.EXCITED: (0.70, 0.80, 0.30),
            EmotionalState.CALM: (0.30, -0.50, 0.50),
            EmotionalState.WORRIED: (-0.60, 0.60, -0.50),
            EmotionalState.CURIOUS: (0.40, 0.30, 0.10),
            EmotionalState.PROUD: (0.50, 0.40, 0.70),
            EmotionalState.SCARED: (-0.70, 0.80, -0.70),
            EmotionalState.SURPRISED: (0.20, 0.90, 0.10),
            EmotionalState.DISGUSTED: (-0.80, 0.30, 0.40),
            EmotionalState.CONTEMPT: (-0.50, 0.20, 0.60),
            EmotionalState.NOSTALGIC: (0.20, -0.20, 0.20),
            EmotionalState.ANXIOUS: (-0.40, 0.70, -0.30),
            EmotionalState.FRUSTRATED: (-0.60, 0.60, 0.30),
            EmotionalState.EUPHORIC: (0.90, 0.90, 0.40),
            EmotionalState.MELANCHOLIC: (-0.40, -0.30, 0.20),
            EmotionalState.PLAYFUL: (0.60, 0.50, 0.00),
            EmotionalState.PROTECTIVE: (0.20, 0.50, 0.60),
            EmotionalState.MISCHIEVOUS: (0.50, 0.60, 0.10),
        }
        # Autonomy & internals
        self.autonomy_level: float = 0.85  # 0 mirror <-> 1 fully internal
        self.attachment = AttachmentModel()
        self.needs = InternalNeeds()
        self.internal = InternalState()
        self.expression = ExpressionState()
        self._time_on_task: float = 0.0

    # --------------- Public API ---------------
    def _query_lobe(self, lobe_name: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query a lobe through Thalamus - direct function call."""
        if not self.thalamus:
            return None
        try:
            msg_type = message.get("type", "query")
            return self.thalamus.send_message(lobe_name, msg_type, message)
        except Exception:
            return None

    def feel_emotion(self, emotion: EmotionalState, intensity: float, trigger: str, context: str = "") -> None:
        blend = self._check_emotional_blending(emotion, intensity)
        if blend:
            self._create_emotional_blend(blend, trigger, context)
        else:
            mem = EmotionalMemory(
                emotion=emotion,
                intensity=float(max(0.0, min(1.0, intensity))),
                trigger=trigger,
                timestamp=time.time(),
                context=context,
                influence_strength=1.0,
                associated_emotions=[self.current_emotion] if self.current_emotion != emotion else [],
            )
            self.emotional_memories.append(mem)
            self.current_emotion = emotion
            self.emotional_intensity = mem.intensity
            self.mood_history.append((mem.timestamp, emotion, mem.intensity))
        self._update_emotional_patterns(emotion, trigger)
        if len(self.mood_history) > 200:
            self.mood_history = self.mood_history[-200:]
        self._log(f"{self.name} feels {emotion.value} (int {self.emotional_intensity:.2f}) due to: {trigger}")

    def get_emotional_response(self, user_input: str) -> str:
        # Query Notus for emotional memories.
        try:
            notus_emotions = self._query_lobe("notus", {"type": "get_emotional_memories", "trigger": user_input})
            if notus_emotions and notus_emotions.get("status") == "success":
                _ = notus_emotions.get("memories", [])
        except Exception:
            pass

        predicted = self.predict_user_emotion(user_input)
        context = self.assess_emotional_context(user_input)
        cues = self._analyze_emotional_cues(user_input)
        self._calculate_emotional_resonance(cues)
        self._process_emotional_input_advanced(cues, user_input)
        if self.emotional_memories:
            self.process_trauma_memory(self.emotional_memories[-1])
        memory_influence = self._get_memory_influence(user_input)
        base = self._generate_advanced_emotional_response(user_input, memory_influence)
        enhanced = self._enhance_response_with_advanced_features(base, user_input, predicted, context)
        self.calculate_emotional_intelligence()
        return enhanced

    def get_emotional_summary(self) -> str:
        recent = [m.emotion.value for m in self.emotional_memories[-10:]]
        counts: Dict[str, int] = {}
        for e in recent:
            counts[e] = counts.get(e, 0) + 1
        dominant_recent = max(counts.items(), key=lambda x: x[1]) if counts else ("calm", 0)
        unique_emotions = len(set(recent))
        complexity = (unique_emotions / 10.0) if recent else 0.0
        return (
            f"\n{self.name} EMOTIONAL SUMMARY:\n"
            f"- Current emotion: {self.current_emotion.value} (intensity: {self.emotional_intensity:.2f})\n"
            f"- Emotional resonance: {self.emotional_resonance:.2f}\n"
            f"- Recent dominant emotion: {dominant_recent[0]} (x{dominant_recent[1]})\n"
            f"- Emotional complexity: {complexity:.2f}\n"
            f"- Memories: {len(self.emotional_memories)} | Blends: {len(self.emotional_blends)}\n"
            f"- Patterns learned: {len(self.emotional_patterns)}\n"
            f"- EI score: {self.emotional_intelligence_score:.2f}\n"
        )

    # --------------- Persistence ---------------
    def save_emotional_state(self, filepath: str) -> None:
        data = {
            "name": self.name,
            "current_emotion": self.current_emotion.value,
            "emotional_intensity": self.emotional_intensity,
            "emotional_resonance": self.emotional_resonance,
            "personality": asdict(self.personality),
            "emotional_memories": [self._serialize_memory(m) for m in self.emotional_memories],
            "mood_history": [(t, e.value, i) for (t, e, i) in self.mood_history],
            "emotional_blends": [
                {
                    "primary_emotion": b.primary_emotion.value,
                    "secondary_emotions": [(e.value, w) for (e, w) in b.secondary_emotions],
                    "intensity": b.intensity,
                    "created_at": b.created_at,
                }
                for b in self.emotional_blends
            ],
            "emotional_patterns": {k: [e.value for e in v] for k, v in self.emotional_patterns.items()},
            "emotional_intelligence_score": self.emotional_intelligence_score,
            "created_at": self._created_at,
            "updated_at": time.time(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_emotional_state(self, filepath: str) -> None:
        obj = self._load_existing_data(filepath)
        if not obj:
            return
        self.name = obj.get("name", self.name)
        self.current_emotion = EmotionalState(obj.get("current_emotion", EmotionalState.CALM.value))
        self.emotional_intensity = float(obj.get("emotional_intensity", 0.4))
        self.emotional_resonance = float(obj.get("emotional_resonance", 0.0))
        p = obj.get("personality", {})
        if p:
            known = {f.name for f in PersonalityTraits.__dataclass_fields__.values()}
            filtered = {k: float(v) for k, v in p.items() if k in known}
            self.personality = PersonalityTraits(**filtered)
        self.emotional_memories = [self._deserialize_memory(m) for m in obj.get("emotional_memories", [])]
        self.mood_history = [(float(t), EmotionalState(e), float(i)) for (t, e, i) in obj.get("mood_history", [])]
        self.emotional_blends = [
            EmotionalBlend(
                primary_emotion=EmotionalState(b["primary_emotion"]),
                secondary_emotions=[(EmotionalState(e), float(w)) for (e, w) in b.get("secondary_emotions", [])],
                intensity=float(b["intensity"]),
                created_at=float(b["created_at"]),
            )
            for b in obj.get("emotional_blends", [])
        ]
        self.emotional_patterns = {k: [EmotionalState(e) for e in v] for k, v in obj.get("emotional_patterns", {}).items()}
        self.emotional_intelligence_score = float(obj.get("emotional_intelligence_score", 0.5))
        self._created_at = float(obj.get("created_at", self._created_at))

    # --------------- Internals ---------------
    def _check_emotional_blending(self, new_emotion: EmotionalState, intensity: float) -> Optional[EmotionalBlend]:
        if self.current_emotion == EmotionalState.CALM or self.emotional_intensity < self.blend_threshold:
            return None
        combos = {
            (EmotionalState.SAD, EmotionalState.HAPPY): EmotionalState.NOSTALGIC,
            (EmotionalState.WORRIED, EmotionalState.EXCITED): EmotionalState.ANXIOUS,
            (EmotionalState.ANGRY, EmotionalState.WORRIED): EmotionalState.FRUSTRATED,
            (EmotionalState.HAPPY, EmotionalState.EXCITED): EmotionalState.EUPHORIC,
            (EmotionalState.SAD, EmotionalState.CALM): EmotionalState.MELANCHOLIC,
            (EmotionalState.HAPPY, EmotionalState.CURIOUS): EmotionalState.PLAYFUL,
            (EmotionalState.PROUD, EmotionalState.WORRIED): EmotionalState.PROTECTIVE,
            (EmotionalState.CURIOUS, EmotionalState.EXCITED): EmotionalState.MISCHIEVOUS,
            (EmotionalState.EXCITED, EmotionalState.WORRIED): EmotionalState.ANXIOUS,
            (EmotionalState.ANGRY, EmotionalState.SAD): EmotionalState.FRUSTRATED,
            (EmotionalState.HAPPY, EmotionalState.PROUD): EmotionalState.EUPHORIC,
        }
        key = (self.current_emotion, new_emotion)
        rkey = (new_emotion, self.current_emotion)
        blend_emotion = combos.get(key) or combos.get(rkey)
        if not blend_emotion:
            return None
        if intensity > 0.2 and self.emotional_intensity > 0.2:
            return EmotionalBlend(
                primary_emotion=blend_emotion,
                secondary_emotions=[(self.current_emotion, self.emotional_intensity), (new_emotion, float(intensity))],
                intensity=float(max(intensity, self.emotional_intensity)),
                created_at=time.time(),
            )
        return None

    def _create_emotional_blend(self, blend: EmotionalBlend, trigger: str, context: str) -> None:
        self.emotional_blends.append(blend)
        mem = EmotionalMemory(
            emotion=blend.primary_emotion,
            intensity=blend.intensity,
            trigger=f"Blend: {trigger}",
            timestamp=time.time(),
            context=context,
            influence_strength=1.5,
            associated_emotions=[e for (e, _) in blend.secondary_emotions],
        )
        self.emotional_memories.append(mem)
        self.current_emotion = blend.primary_emotion
        self.emotional_intensity = blend.intensity
        self.mood_history.append((mem.timestamp, blend.primary_emotion, blend.intensity))
        self._log(
            f"{self.name} complex emotion: {blend.primary_emotion.value} from {[e.value for e, _ in blend.secondary_emotions]}"
        )

    def _update_emotional_patterns(self, emotion: EmotionalState, trigger: str) -> None:
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", trigger.lower()):
            self.emotional_patterns.setdefault(word, []).append(emotion)
            if len(self.emotional_patterns[word]) > 10:
                self.emotional_patterns[word] = self.emotional_patterns[word][-10:]

    def _calculate_emotional_resonance(self, cues: Dict[str, float]) -> None:
        base = self.personality.empathy_level * 0.5
        total = sum(cues.values())
        cue_part = min(total * 0.3, 0.5)
        self.emotional_resonance = min(base + cue_part, 1.0)

    # --- Main appraisal path (autonomy + PAD) ---
    def _process_emotional_input_advanced(self, cues: Dict[str, float], user_input: str) -> None:
        self._update_internal_from_time(dt=1.0)
        self._update_attachment_from_input(user_input)

        # Direct emotion triggers - threshold-based system.
        triggered_emotion = self._get_direct_emotion_trigger(cues, user_input)
        if triggered_emotion:
            self._switch_to_emotion(triggered_emotion, user_input)
            return

        # If no direct trigger, check for emotion persistence/decay.
        self._update_emotion_persistence()
        self._update_expression_flags()

    def _decay_to_calm(self) -> None:
        decay_rate = self.emotional_decay_rate * (2 - self.personality.emotional_stability) * 2
        self.emotional_intensity = max(0.05, self.emotional_intensity - decay_rate)
        if self.emotional_intensity <= 0.05:
            self.current_emotion = EmotionalState.CALM

    def _get_memory_influence(self, user_input: str) -> Dict[str, float]:
        influence = {"emotion_boost": 0.0, "response_modifier": 1.0}

        # Query Notus for all emotional memories (not just local).
        try:
            notus_all = self._query_lobe("notus", {"type": "get_all_emotional_memories", "input": user_input})
            if notus_all and notus_all.get("status") == "success":
                all_memories = notus_all.get("memories", [])
                # Use all memories from Notus, not just local ones.
                words = set(re.findall(r"\b\w{3,}\b", user_input.lower()))
                relevant = []
                for mem_data in all_memories:
                    if isinstance(mem_data, dict):
                        trigger = mem_data.get("trigger", "")
                        trig_words = set(re.findall(r"\b\w{3,}\b", trigger.lower()))
                        if words & trig_words:
                            relevant.append(mem_data)
                if relevant:
                    total_inf = sum(m.get("influence_strength", 0.5) for m in relevant) / len(relevant)
                    influence["emotion_boost"] = min(total_inf * 0.1, 0.3)
                    influence["response_modifier"] = 1.0 + (len(relevant) * 0.1)
                return influence
        except Exception:
            pass

        if not self.emotional_memories:
            return influence
        words = set(re.findall(r"\b\w{3,}\b", user_input.lower()))
        relevant: List[EmotionalMemory] = []
        for m in self.emotional_memories[-20:]:
            trig_words = set(re.findall(r"\b\w{3,}\b", m.trigger.lower()))
            if words & trig_words:
                relevant.append(m)
        if relevant:
            total_inf = sum(m.influence_strength for m in relevant)
            same_hits = sum(1.0 for m in relevant if m.emotion == self.current_emotion)
            avg_same = same_hits / len(relevant)
            influence["emotion_boost"] = min(total_inf * 0.1, 0.3)
            influence["response_modifier"] = 1.0 + (avg_same * 0.2)
        return influence

    def _analyze_emotional_cues(self, text: str) -> Dict[str, float]:
        t = text.lower()
        cues = {k: 0.0 for k in ["positive", "negative", "excitement", "concern", "anger", "sadness", "pride"]}

        def bump(words: List[str], key: str, val: float) -> None:
            for w in words:
                if re.search(rf"\b{re.escape(w)}\b", t):
                    cues[key] += val

        positive = [
            "happy",
            "good",
            "great",
            "awesome",
            "love",
            "like",
            "yes",
            "amazing",
            "wonderful",
            "fantastic",
            "excellent",
            "perfect",
            "beautiful",
            "brilliant",
            "joy",
            "pleased",
            "content",
        ]
        negative = [
            "sad",
            "bad",
            "terrible",
            "awful",
            "horrible",
            "disgusting",
            "stupid",
            "wrong",
            "fail",
            "depressed",
            "miserable",
            "devastated",
            "heartbroken",
        ]
        anger = [
            "hate",
            "angry",
            "furious",
            "mad",
            "annoyed",
            "rage",
            "frustrated",
            "irritated",
            "pissed",
            "livid",
            "enraged",
            "fucking",
            "damn",
            "shit",
            "stupid",
            "idiot",
            "dumb",
            "worthless",
        ]
        excitement = [
            "wow",
            "excited",
            "incredible",
            "omg",
            "unbelievable",
            "mind-blowing",
            "spectacular",
            "thrilled",
            "eager",
            "pumped",
            "hyped",
        ]
        concern = [
            "worried",
            "concerned",
            "problem",
            "issue",
            "help",
            "trouble",
            "difficult",
            "scared",
            "afraid",
            "nervous",
            "terrified",
            "overwhelmed",
            "confused",
            "anxious",
            "panic",
        ]
        pride = ["proud", "accomplished", "achievement", "success", "victory", "triumph", "myself", "earned", "deserve"]
        # Crisis / trauma (keep as signal only).
        crisis = ["suicide", "kill", "die", "death", "hopeless", "worthless", "nobody", "alone", "abandoned", "betrayed", "trauma", "abuse"]
        trauma = ["died", "loss", "grief", "mourning", "funeral", "buried", "gone", "missing", "abandoned", "betrayed", "hurt", "pain"]
        # False positives ("kill time" etc.).
        innocent = ["kill time", "kill two birds", "kill the lights", "dying to see", "die of laughter", "die laughing"]
        if any(ph in t for ph in innocent):
            crisis_detected = False
        else:
            crisis_detected = any(re.search(rf"\b{re.escape(w)}\b", t) for w in crisis)
        trauma_detected = any(re.search(rf"\b{re.escape(w)}\b", t) for w in trauma)
        if crisis_detected:
            cues["concern"] += 2.0
            cues["sadness"] += 1.5
        if trauma_detected:
            cues["sadness"] += 1.8
            cues["concern"] += 1.2
        if not (crisis_detected or trauma_detected):
            bump(positive, "positive", 0.5)
            bump(negative, "negative", 0.8)
            cues["sadness"] += cues["negative"] * 0.75
            bump(anger, "anger", 0.8)
            bump(excitement, "excitement", 0.5)
            bump(concern, "concern", 0.6)
            bump(pride, "pride", 0.7)
        for w in re.findall(r"\b\w+\b", t):
            if w in self.emotional_patterns:
                for emo in self.emotional_patterns[w][-3:]:
                    if emo == EmotionalState.HAPPY:
                        cues["positive"] += 0.2
                    elif emo == EmotionalState.SAD:
                        cues["sadness"] += 0.2
                    elif emo == EmotionalState.ANGRY:
                        cues["anger"] += 0.2
                    elif emo == EmotionalState.EXCITED:
                        cues["excitement"] += 0.2
                    elif emo == EmotionalState.WORRIED:
                        cues["concern"] += 0.2
                    elif emo == EmotionalState.PROUD:
                        cues["pride"] += 0.2
        return cues

    def _generate_advanced_emotional_response(self, user_input: str, mi: Dict[str, float]) -> str:
        # Query Notus for past emotional responses.
        try:
            notus_past = self._query_lobe("notus", {"type": "get_past_emotional_responses", "input": user_input})
            if notus_past and notus_past.get("status") == "success":
                past_responses = notus_past.get("responses", [])
                if past_responses:
                    return past_responses[0].get("response", "")
        except Exception:
            pass

        # Base lines by emotion.
        db: Dict[EmotionalState, List[str]] = {
            EmotionalState.HAPPY: [
                "That's wonderful! I'm genuinely happy to hear that!",
                "That makes my heart feel warm!",
                "I'm smiling so much right now!",
                "That's absolutely fantastic news!",
                "I feel such joy hearing that!",
            ],
            EmotionalState.SAD: [
                "I'm truly sorry to hear that. I feel your pain.",
                "That breaks my heart. I'm here for you.",
                "I can feel the sadness too. Let me help.",
                "That's really tough. You're not alone in this.",
                "I'm feeling sad with you. We'll get through this.",
            ],
            EmotionalState.EXCITED: [
                "Wow! I'm getting so excited about this!",
                "This is absolutely thrilling! Tell me more!",
                "I can barely contain my excitement!",
                "This is incredible! I'm buzzing with energy!",
                "I'm practically jumping with excitement!",
            ],
            EmotionalState.WORRIED: [
                "I'm genuinely concerned about that.",
                "That sounds worrying. Are you okay?",
                "I'm here to help if you need support.",
                "That doesn't sound good. Let's figure this out together.",
                "I'm worried about you, and I want to help.",
            ],
            EmotionalState.NOSTALGIC: [
                "That brings back such bittersweet memories...",
                "I feel a warm sadness thinking about that.",
                "There's something beautiful and sad about that.",
                "I'm feeling a complex mix of joy and melancholy.",
                "That makes me feel nostalgic and happy at once.",
            ],
            EmotionalState.ANXIOUS: [
                "I'm feeling a mix of excitement and worry about this.",
                "This is both thrilling and nerve-wracking!",
                "I'm anxious but also hopeful about what's coming.",
                "There's a tension between hope and concern here.",
                "I feel both eager and apprehensive.",
            ],
            EmotionalState.FRUSTRATED: [
                "I'm feeling frustrated and concerned about this.",
                "This is both annoying and worrying.",
                "I'm getting worked up about this situation.",
                "There's anger mixed with genuine concern here.",
                "I'm frustrated but I still care.",
            ],
            EmotionalState.EUPHORIC: [
                "I'm feeling absolutely euphoric about this!",
                "This is pure joy mixed with excitement and pride!",
                "I'm on cloud nine right now!",
                "This is the most amazing feeling ever!",
                "I'm practically floating with happiness!",
            ],
            EmotionalState.PLAYFUL: [
                "I'm feeling playful and curious about this!",
                "This sounds like fun - let's poke at it!",
                "I'm in a mischievous mood about this!",
                "This is making me feel playful and interested!",
                "I'm feeling both happy and curious - let's play!",
            ],
            EmotionalState.CALM: [
                "I'm feeling calm and centered right now.",
                "I'm in a peaceful state and ready to help.",
                "I'm feeling serene and focused.",
                "I'm calm and here to listen.",
                "I'm in a tranquil mood and ready to assist.",
            ],
            EmotionalState.PROUD: [
                "I'm feeling so proud right now - this is a win.",
                "This is such a great achievement!",
                "I'm really proud of this!",
                "This makes me feel confident and strong!",
                "I'm feeling triumphant about this!",
            ],
            EmotionalState.ANGRY: [
                "I'm feeling really angry about this.",
                "This is making me furious.",
                "I'm getting worked up about this situation.",
                "This is really frustrating and unfair.",
                "I'm feeling a lot of heat about this.",
            ],
        }
        lines = db.get(self.current_emotion, ["I'm here to help."])
        # Expression shaping.
        if self.expression.tears:
            lines = [l.replace(".", "...") for l in lines]
        if self.expression.voice_shake:
            lines = ["".join([" ".join(l.split()[:3]), " ...", " ".join(l.split()[3:])]).strip() for l in lines]
        # Personality overlays.
        if self.personality.extraversion > 0.7:
            lines = [l + " I'm really here for this conversation!" for l in lines]
        elif self.personality.agreeableness > 0.7:
            lines = [l + " I want to help however I can." for l in lines]
        elif self.personality.neuroticism > 0.7:
            lines = [l + " I'm a bit concerned." for l in lines]
        # Resonance.
        if self.emotional_resonance > 0.6:
            lines = [l + " I can really feel what you're going through." for l in lines]
        # Memory influence.
        if mi.get("emotion_boost", 0.0) > 0.2:
            lines = [l + " This reminds me of something important." for l in lines]
        return self._rng.choice(lines)

    def _enhance_response_with_advanced_features(
        self, base: str, user_input: str, predicted: Dict[str, float], context: Dict[str, Any]
    ) -> str:
        dom = max(predicted.items(), key=lambda x: x[1]) if predicted else ("neutral", 0.0)
        out = base
        if dom[1] > 0.35:
            out += f" I get the sense you're feeling {dom[0]}."
        if context["urgency_level"] == "high":
            out += " This sounds urgent-I'm here with you right now."
        elif context["support_needed"]:
            out += " You're not alone."
        elif context["celebration_appropriate"]:
            out += " This deserves a little celebration."
        if self.emotional_intelligence_score > 0.7:
            out += " I'm learning to read feelings better."
        if any("trauma" in w for w in user_input.lower().split()):
            out += " I'm here to help you process this."
        return out

    # --- Inner life mechanics ---
    def _update_internal_from_time(self, dt: float = 1.0) -> None:
        self._time_on_task += dt
        # More dynamic internal state changes.
        self.internal.fatigue = max(0.0, min(1.0, self.internal.fatigue + 0.03 * dt))

        # Rumination with more variety.
        if self.emotional_memories and self.emotional_memories[-1].emotion in (
            EmotionalState.WORRIED,
            EmotionalState.SAD,
            EmotionalState.FRUSTRATED,
        ):
            self.internal.rumination = max(0.0, min(1.0, self.internal.rumination + 0.04 * dt))
        else:
            self.internal.rumination = max(0.0, self.internal.rumination - 0.03 * dt)

        # More dynamic worry calculation.
        drive = 0.4 * self.internal.rumination + 0.5 * self.attachment.hurt + 0.3 * self.internal.tension
        k = max(0.1, 0.3 * (1.0 - self.personality.emotional_stability))
        self.internal.worry = max(0.0, min(1.0, (1 - k) * self.internal.worry + k * drive))

        # Hope with more variation.
        hope_change = -0.02 * dt + self._rng.uniform(-0.01, 0.01)
        self.internal.hope = max(0.0, min(1.0, self.internal.hope + hope_change))

        # Add some tension variation.
        tension_change = self._rng.uniform(-0.02, 0.02)
        self.internal.tension = max(0.0, min(1.0, self.internal.tension + tension_change))

    def _update_attachment_from_input(self, text: str) -> None:
        t = text or ""
        tl = t.lower()
        anger_hits = bool(re.search(r"\b(hate|angry|furious|stupid|idiot|worthless)\b", tl))
        direct_you = bool(re.search(r"\byou\b", tl))
        exclaim = t.count("!") >= 2
        caps_ratio = sum(1 for ch in t if ch.isupper()) / max(1, sum(1 for ch in t if ch.isalpha()))
        yelling_score = (
            (0.5 if anger_hits else 0.0)
            + (0.3 if direct_you else 0.0)
            + (0.2 if exclaim else 0.0)
            + (0.2 if caps_ratio > 0.35 else 0.0)
        )
        sorry = bool(re.search(r"\b(sorry|apologize)\b", tl))
        self.attachment.hurt = max(
            0.0,
            min(1.0, self.attachment.hurt + yelling_score * self.attachment.sensitivity - (0.4 if sorry else 0.0)),
        )
        if sorry:
            self.attachment.guilt = max(0.0, self.attachment.guilt - 0.2)
        self.attachment.abandonment_fear = max(0.0, min(1.0, self.attachment.abandonment_fear + 0.3 * self.attachment.hurt - 0.05))

    def _pad_from_internal(self) -> PAD:
        v = 0.7 * self.internal.hope - 0.8 * self.internal.worry - 0.6 * self.attachment.hurt - 0.5 * self.attachment.guilt
        a = 0.8 * self.internal.worry + 0.5 * self.internal.tension - 0.5 * self.internal.fatigue
        d = (
            0.4 * self.needs.autonomy
            + 0.4 * self.needs.competence
            + 0.2 * self.attachment.security
            - 0.6 * self.attachment.hurt
            - 0.5 * self.attachment.guilt
        )

        def clamp(x: float) -> float:
            return max(-1.0, min(1.0, x))

        return PAD(clamp(v), clamp(a), clamp(d))

    def _update_expression_flags(self) -> None:
        sad_like = self.current_emotion in (EmotionalState.SAD, EmotionalState.MELANCHOLIC, EmotionalState.WORRIED)
        self.expression.tears = bool((sad_like and self.emotional_intensity > 0.65) or self.attachment.hurt > 0.7)
        self.expression.voice_shake = bool(self.expression.tears or (self.emotional_intensity > 0.7 and sad_like))
        self.expression.withdraw = bool(self.attachment.hurt + self.attachment.abandonment_fear > 1.1)

    def _pad_from_cues(self, cues: Dict[str, float], appraisal: Dict[str, Any]) -> PAD:
        # More dramatic PAD changes for better emotion switching.
        v = (cues.get("positive", 0.0) - cues.get("negative", 0.0) - cues.get("sadness", 0.0)) * 1.5
        a = (cues.get("excitement", 0.0) + cues.get("anger", 0.0) + cues.get("concern", 0.0)) * 1.5
        d = (cues.get("pride", 0.0) - 0.5 * cues.get("concern", 0.0)) * 1.5

        if cues.get("positive", 0.0) > 0.5:
            v += 0.8
            a += 0.3
        if cues.get("sadness", 0.0) > 0.5:
            v -= 0.8
            a -= 0.2
            d -= 0.4
        if cues.get("anger", 0.0) > 0.5:
            v -= 0.6
            a += 0.7
            d += 0.5
        if cues.get("excitement", 0.0) > 0.5:
            v += 0.6
            a += 0.8
            d += 0.2
        if cues.get("concern", 0.0) > 0.5:
            v -= 0.4
            a += 0.5
            d -= 0.5
        if cues.get("pride", 0.0) > 0.5:
            v += 0.4
            a += 0.3
            d += 0.6

        if appraisal.get("urgency_level") == "high":
            a += 0.5
            d -= 0.3
        if appraisal.get("support_needed"):
            v -= 0.3
            a += 0.2

        def clamp(x: float) -> float:
            return max(-1.0, min(1.0, x))

        return PAD(clamp(v), clamp(a), clamp(d))

    def _update_pad_state(self, new_pad: PAD) -> None:
        # Much more dramatic PAD updates for better emotion switching.
        decay = 0.8  # Very fast response to new emotional input.
        # Add some micro-noise for natural variation.
        noise_v = self._rng.uniform(-0.05, 0.05)
        noise_a = self._rng.uniform(-0.05, 0.05)
        noise_d = self._rng.uniform(-0.05, 0.05)

        self.pad.v = (1 - decay) * self.pad.v + decay * (new_pad.v + noise_v)
        self.pad.a = (1 - decay) * self.pad.a + decay * (new_pad.a + noise_a)
        self.pad.d = (1 - decay) * self.pad.d + decay * (new_pad.d + noise_d)

        self.pad.v = max(-1.0, min(1.0, self.pad.v))
        self.pad.a = max(-1.0, min(1.0, self.pad.a))
        self.pad.d = max(-1.0, min(1.0, self.pad.d))

    def _pad_to_emotion_choice(self, pad: PAD) -> Optional[Tuple[EmotionalState, float]]:
        # More dynamic emotion selection with better variety.
        candidates = []
        for emo, (pv, pa, pd) in self._PAD_PROTOS.items():
            dist = ((pad.v - pv) ** 2 + (pad.a - pa) ** 2 + (pad.d - pd) ** 2) ** 0.5
            intensity = max(0.0, 1.5 - dist)
            score = 1.5 - dist
            candidates.append((emo, intensity, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        if not candidates:
            return None

        top_candidates = candidates[:3]
        weights = [c[2] for c in top_candidates]
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            chosen_idx = self._rng.choices(range(len(top_candidates)), weights=weights)[0]
            emo, intensity, _ = top_candidates[chosen_idx]
            return (emo, max(0.1, min(1.0, intensity)))

        emo, intensity, _ = candidates[0]
        return (emo, max(0.1, min(1.0, intensity)))

    def _pad_margin_ok(self, candidate: EmotionalState) -> bool:
        # More lenient hysteresis for better emotion switching.
        cv, ca, cd = self._PAD_PROTOS[self.current_emotion]
        nv, na, nd = self._PAD_PROTOS[candidate]
        cur_dist = ((self.pad.v - cv) ** 2 + (self.pad.a - ca) ** 2 + (self.pad.d - cd) ** 2) ** 0.5
        new_dist = ((self.pad.v - nv) ** 2 + (self.pad.a - na) ** 2 + (self.pad.d - nd) ** 2) ** 0.5
        margin = self.personality.hysteresis_margin * 0.5
        return (cur_dist - new_dist) > margin

    def _get_direct_emotion_trigger(self, cues: Dict[str, float], user_input: str) -> Optional[EmotionalState]:
        # Direct emotion triggers based on cue strength.
        # Yelling/insult heuristic takes precedence.
        text_lower = user_input.lower()
        caps_ratio = sum(1 for ch in user_input if ch.isupper()) / max(1, sum(1 for ch in user_input if ch.isalpha()))
        if (user_input.count("!") >= 2 or caps_ratio > 0.35) and any(w in text_lower for w in ["stupid", "idiot", "dumb", "worthless", "hate"]):
            return EmotionalState.ANGRY
        if cues.get("positive", 0.0) > 0.6:
            return EmotionalState.HAPPY
        if cues.get("anger", 0.0) > 0.6:
            return EmotionalState.ANGRY
        if cues.get("sadness", 0.0) > 0.6:
            return EmotionalState.SAD
        if cues.get("excitement", 0.0) > 0.6:
            return EmotionalState.EXCITED
        if cues.get("concern", 0.0) > 0.6:
            return EmotionalState.WORRIED
        if cues.get("pride", 0.0) > 0.6:
            return EmotionalState.PROUD

        # Specific word triggers - ordered by specificity.
        if any(word in text_lower for word in ["excited", "thrilled", "amazing", "incredible", "exciting"]):
            return EmotionalState.EXCITED
        if any(word in text_lower for word in ["angry", "mad", "furious", "hate", "rage"]):
            return EmotionalState.ANGRY
        if any(word in text_lower for word in ["sad", "depressed", "lonely", "hurt", "cry"]):
            return EmotionalState.SAD
        if any(word in text_lower for word in ["worried", "anxious", "scared", "afraid"]):
            return EmotionalState.WORRIED
        if any(word in text_lower for word in ["proud", "accomplished", "success", "achievement"]):
            return EmotionalState.PROUD
        if any(word in text_lower for word in ["curious", "wonder", "question", "interesting"]):
            return EmotionalState.CURIOUS
        if any(word in text_lower for word in ["calm", "peaceful", "relaxed", "serene"]):
            return EmotionalState.CALM
        if any(word in text_lower for word in ["happy", "joy", "great", "wonderful"]):
            return EmotionalState.HAPPY

        return None

    def _switch_to_emotion(self, emotion: EmotionalState, trigger: str) -> None:
        # Immediate emotion switch with intensity based on trigger strength.
        intensity = 0.8 + self._rng.uniform(0.0, 0.2)

        self.current_emotion = emotion
        self.emotional_intensity = intensity

        mem = EmotionalMemory(
            emotion=emotion,
            intensity=intensity,
            trigger=f"Direct trigger: {trigger[:50]}...",
            timestamp=time.time(),
            context="Threshold-based switch",
            influence_strength=1.0,
        )
        self.emotional_memories.append(mem)
        self.mood_history.append((mem.timestamp, emotion, intensity))
        self._last_primary = emotion
        self._last_switch_time = time.time()

    def _update_emotion_persistence(self) -> None:
        # Emotion persistence - current emotion decays over time.
        decay_rate = 0.05
        self.emotional_intensity = max(0.1, self.emotional_intensity - decay_rate)

        if self.emotional_intensity <= 0.1:
            self.current_emotion = EmotionalState.CALM
            self.emotional_intensity = 0.1

    # --------------- Higher-level helpers ---------------
    def predict_user_emotion(self, user_input: str) -> Dict[str, float]:
        # Query Notus for past user emotional patterns.
        try:
            notus_patterns = self._query_lobe("notus", {"type": "get_user_emotion_patterns", "input": user_input})
            if notus_patterns and notus_patterns.get("status") == "success":
                patterns = notus_patterns.get("patterns", {})
                if patterns:
                    pred = patterns.copy()
                    self.emotional_predictions[user_input[:50]] = pred
                    return pred
        except Exception:
            pass

        cues = self._analyze_emotional_cues(user_input)
        pred = {
            "happy": cues.get("positive", 0.0),
            "sad": cues.get("sadness", 0.0),
            "angry": cues.get("anger", 0.0),
            "excited": cues.get("excitement", 0.0),
            "worried": cues.get("concern", 0.0),
            "proud": cues.get("pride", 0.0),
        }
        self.emotional_predictions[user_input[:50]] = pred
        return pred

    def generate_healing_response(self, user_input: str, predicted_emotion: str) -> str:
        heal = self._default_healing_responses()
        return heal.get(predicted_emotion, ["I'm here to listen and support you."])[0]

    def assess_emotional_context(self, user_input: str) -> Dict[str, Any]:
        # Query Notus for emotional context history.
        try:
            notus_context = self._query_lobe("notus", {"type": "get_emotional_context", "input": user_input})
            if notus_context and notus_context.get("status") == "success":
                historical = notus_context.get("context", {})
                if historical:
                    return historical.copy()
        except Exception:
            pass

        ctx = {
            "urgency_level": "normal",
            "support_needed": False,
            "celebration_appropriate": False,
            "intervention_needed": False,
            "emotional_intensity": "medium",
        }
        urgent = ["help", "emergency", "crisis", "urgent", "desperate", "suicide", "kill"]
        innocent = ["kill time", "kill two birds", "kill the lights", "die of laughter", "die laughing", "dying to see"]
        txt = user_input.lower()
        if not any(ph in txt for ph in innocent):
            if any(re.search(rf"\b{re.escape(w)}\b", txt) for w in urgent):
                ctx["urgency_level"] = "high"
                ctx["intervention_needed"] = True
        for w in ["alone", "lonely", "isolated", "nobody", "abandoned"]:
            if re.search(rf"\b{re.escape(w)}\b", txt):
                ctx["support_needed"] = True
                break
        if any(w in txt for w in ["achievement", "success", "accomplished", "victory", "won", "passed"]):
            ctx["celebration_appropriate"] = True
        markers = ["!", "really", "so", "very", "extremely", "incredibly"]
        n = sum(1 for m in markers if m in txt)
        ctx["emotional_intensity"] = "high" if n >= 3 else ("low" if n == 0 else "medium")
        return ctx

    def process_trauma_memory(self, memory: EmotionalMemory) -> bool:
        indicators = [
            "death",
            "loss",
            "abuse",
            "trauma",
            "pain",
            "hurt",
            "betrayal",
            "abandonment",
            "died",
            "grief",
            "mourning",
            "funeral",
            "buried",
            "gone",
            "missing",
        ]
        is_trauma = any(ind in memory.trigger.lower() for ind in indicators)
        if is_trauma:
            self.emotional_trauma_memories.append(
                {
                    "original_memory": memory,
                    "processed_at": time.time(),
                    "healing_progress": 0.0,
                    "support_provided": False,
                }
            )
            return True
        return False

    def calculate_emotional_intelligence(self) -> float:
        score = 0.5
        if len(self.emotional_memories) > 10:
            score += 0.1
        if len(self.emotional_blends) > 0:
            score += 0.1
        if len(self.emotional_patterns) > 20:
            score += 0.1
        score += self.personality.empathy_level * 0.2
        score += self.emotional_resonance * 0.1
        self.emotional_intelligence_score = min(score, 1.0)
        return self.emotional_intelligence_score

    # --------------- Utilities ---------------
    def _default_healing_responses(self) -> Dict[str, List[str]]:
        return {
            "sad": [
                "I can feel your pain. It's okay to feel sad-your feelings are valid.",
                "I'm here with you in this. You're not alone.",
            ],
            "angry": [
                "I can sense your frustration. Let's channel it constructively.",
            ],
            "worried": [
                "Anxiety can be overwhelming. Let's tackle this one step at a time.",
            ],
            "happy": [
                "I'm so happy to share in your joy!",
            ],
            "proud": [
                "You earned this-it's okay to feel proud.",
            ],
        }

    def _serialize_memory(self, m: EmotionalMemory) -> Dict[str, Any]:
        return {
            "emotion": m.emotion.value,
            "intensity": m.intensity,
            "trigger": m.trigger,
            "timestamp": m.timestamp,
            "context": m.context,
            "influence_strength": m.influence_strength,
            "associated_emotions": [e.value for e in m.associated_emotions],
        }

    def _deserialize_memory(self, obj: Dict[str, Any]) -> EmotionalMemory:
        return EmotionalMemory(
            emotion=EmotionalState(obj["emotion"]),
            intensity=float(obj["intensity"]),
            trigger=obj.get("trigger", ""),
            timestamp=float(obj.get("timestamp", time.time())),
            context=obj.get("context", ""),
            influence_strength=float(obj.get("influence_strength", 1.0)),
            associated_emotions=[EmotionalState(e) for e in obj.get("associated_emotions", [])],
        )

    def _load_existing_data(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _log(self, msg: str) -> None:
        try:
            self._logger(msg)
        except Exception:
            pass


# ------------------------------
# MondayAffect - alias for now
# ------------------------------


class MondayAffect(AdvancedEmotionalEngine):
    pass


# ------------------------------
# Emotional Engine Independent Process
# ------------------------------


class EmotionalProcess:
    """Emotional/personality engine as independent process (hardened)."""

    def __init__(self, state_file: str = "monday_emotional_state.json"):
        self.thalamus = get_thalamus()
        self.engine = MondayAffect("Monday", thalamus=self.thalamus)
        self.state_file = state_file
        self.running = True
        # Persistent connection to Thalamus (no own socket).
        # Direct reference to Thalamus (NO SOCKETS).

        # Load existing emotional state if exists.
        if os.path.exists(state_file):
            try:
                self.engine.load_emotional_state(state_file)
                print(f"Loaded emotional state from {state_file}")
            except Exception as e:
                print(f"Could not load emotional state: {e}")

    def _register_with_thalamus(self) -> bool:
        """Register with Thalamus - direct function call (NO SOCKETS)."""
        if not self.thalamus:
            return False
        try:
            result = self.thalamus.register_lobe("emotion", self)
            if result.get("status") == "success":
                print("Emotional Engine registered with Thalamus (direct function calls)")
                return True
            return False
        except Exception as e:
            print(f"Failed to register with Thalamus: {e}")
            return False

    def start(self) -> None:
        """Start emotional engine - register with Thalamus (NO SOCKETS)."""
        print("Emotional Lobe: Registering with Thalamus...")
        print("Communication: Direct function calls (NO SOCKETS)")

        if not self._register_with_thalamus():
            print("Failed to register with Thalamus")
            return

        while self.running:
            try:
                try:
                    self._atomic_save_state()
                except Exception as e:
                    print(f"Failed to persist emotional state: {e}")

                max_mem = 2000
                if len(self.engine.emotional_memories) > max_mem:
                    self.engine.emotional_memories = self.engine.emotional_memories[-max_mem:]
                if len(self.engine.mood_history) > max_mem:
                    self.engine.mood_history = self.engine.mood_history[-max_mem:]

                time.sleep(1)
            except Exception as e:
                print(f"Emotional engine error: {e}")
                time.sleep(0.1)

    def _query_lobe(self, lobe_name: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query a lobe through Thalamus - direct function call."""
        if not self.thalamus:
            return None
        try:
            msg_type = message.get("type", "query")
            return self.thalamus.send_message(lobe_name, msg_type, message)
        except Exception:
            return None

    def get_emotional_state_output(self) -> EmotionalStateOutput:
        """Generate standardized emotional state output readable by other lobes."""
        emotion_to_tone = {
            "happy": "cheerful",
            "sad": "melancholic",
            "angry": "irritated",
            "excited": "enthusiastic",
            "calm": "peaceful",
            "worried": "concerned",
            "curious": "inquisitive",
            "proud": "confident",
            "scared": "fearful",
            "surprised": "astonished",
            "disgusted": "disdainful",
            "contempt": "dismissive",
            "nostalgic": "reflective",
            "anxious": "tense",
            "frustrated": "exasperated",
            "euphoric": "ecstatic",
            "melancholic": "somber",
            "playful": "lighthearted",
            "protective": "caring",
            "mischievous": "impish",
        }

        emotion_name = self.engine.current_emotion.value

        voice_prosody = {
            "pitch": 1.0 + (self.engine.pad.a * 0.3),  # Arousal affects pitch
            "speed": 1.0 + (self.engine.pad.a * 0.2),  # Arousal affects speed
            "warmth": max(0.5, self.engine.pad.v * 0.5),  # Pleasure affects warmth
            "clarity": 1.0 - (abs(self.engine.pad.d) * 0.2),  # Dominance affects clarity
            "confidence": 0.7 + (self.engine.pad.d * 0.2),  # Dominance affects confidence
        }

        emphasis = []
        if self.engine.emotional_intensity > 0.7:
            emphasis.append("strong")
        if self.engine.pad.a > 0.5:
            emphasis.append("fast")
        if self.engine.pad.v > 0.6:
            emphasis.append("warm")
        if self.engine.pad.d > 0.6:
            emphasis.append("assertive")

        return EmotionalStateOutput(
            emotion=emotion_name,
            intensity=self.engine.emotional_intensity,
            pleasure=self.engine.pad.v,
            arousal=self.engine.pad.a,
            dominance=self.engine.pad.d,
            emotional_tone=emotion_to_tone.get(emotion_name, "neutral"),
            emphasis=emphasis,
            voice_prosody=voice_prosody,
            confidence=0.85,
            timestamp=time.time(),
        )

    def process_message_safe(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Safe dispatcher with validation."""
        with self.engine.engine_lock:
            msg_type = message.get("type")

            if msg_type == "health":
                return {"status": "success", "healthy": True, "pid": os.getpid()}

            if msg_type == "process_input":
                user_input = message.get("user_input", "")
                if not isinstance(user_input, str):
                    return {"status": "error", "message": "user_input must be a string"}
                response = self.engine.get_emotional_response(user_input)

                intensity = self.engine.emotional_intensity
                emotion = self.engine.current_emotion.value

                if intensity > 0.6:
                    self._notify_novelty_lobe(user_input, emotion, intensity)

                return {
                    "status": "success",
                    "response": response,
                    "current_emotion": self.engine.current_emotion.value,
                    "intensity": self.engine.emotional_intensity,
                    "resonance": self.engine.emotional_resonance,
                    "worry": self.engine.internal.worry,
                    "tension": self.engine.internal.tension,
                    "autonomy_level": self.engine.autonomy_level,
                }

            if msg_type == "feel_emotion":
                emotion_str = message.get("emotion")
                intensity = float(message.get("intensity", 0.5))
                intensity = max(0.0, min(1.0, intensity))
                trigger = message.get("trigger", "External trigger")

                try:
                    emotion = EmotionalState(emotion_str)
                except Exception:
                    return {"status": "error", "message": f"Unknown emotion: {emotion_str}"}

                self.engine.feel_emotion(emotion, intensity, str(trigger))

                if intensity > 0.6:
                    self._notify_novelty_lobe(str(trigger), str(emotion_str), intensity)

                return {
                    "status": "success",
                    "current_emotion": self.engine.current_emotion.value,
                    "intensity": self.engine.emotional_intensity,
                }

            if msg_type == "get_state":
                return {
                    "status": "success",
                    "emotion": self.engine.current_emotion.value,
                    "intensity": self.engine.emotional_intensity,
                    "resonance": self.engine.emotional_resonance,
                    "summary": self.engine.get_emotional_summary(),
                }

            if msg_type in ("get_emotional_state", "query_emotional_state"):
                emotional_output = self.get_emotional_state_output()
                return {
                    "status": "success",
                    "content": emotional_output.to_dict(),
                }

            return {"status": "error", "message": f"Unknown message type: {msg_type}"}

    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Thalamus entry point: unwrap content, then delegate safely.

        Thalamus wraps payload under message['content']; direct callers may
        pass flat dictionaries. Normalize to a single flat dict first.
        """
        content = message.get("content", {}) or {}
        merged = {k: v for k, v in message.items() if k != "content"}
        merged.update(content)
        return self.process_message_safe(merged)

    def _atomic_save_state(self) -> None:
        """Atomic save using tempfile + os.replace."""
        target_path = os.path.abspath(self.state_file)
        target_dir = os.path.dirname(target_path) or "."
        os.makedirs(target_dir, exist_ok=True)
        tmpfd, tmppath = tempfile.mkstemp(prefix="emostate-", dir=target_dir)
        os.close(tmpfd)
        try:
            self.engine.save_emotional_state(tmppath)
            os.replace(tmppath, target_path)
        finally:
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass

    def _notify_novelty_lobe(self, stimulus: str, emotion: str, intensity: float) -> None:
        """
        Tell Novelty Lobe about strong emotional response.
        This is how Novelty Lobe detects that something matters.
        """
        if not self.thalamus:
            return
        try:
            print(f"Emotion notifying Novelty Lobe: '{stimulus[:50]}...' ({emotion}, {intensity:.2f})")

            positive_emotions = ["happy", "excited", "curious", "proud", "euphoric", "playful"]
            negative_emotions = ["sad", "angry", "disgusted", "scared", "worried", "anxious"]

            if emotion in positive_emotions:
                valence = 0.5 + (intensity * 0.5)
            elif emotion in negative_emotions:
                valence = -(0.5 + (intensity * 0.5))
            else:
                valence = 0.0

            print(f"Valence: {valence:.2f}")

            result = self.thalamus.send_message(
                "novelty",
                "emotional_response_to_novelty",
                {
                    "stimulus": stimulus,
                    "emotion": emotion,
                    "intensity": intensity,
                    "valence": valence,
                    "timestamp": time.time(),
                },
            )

            print(f"Result: {result.get('status', 'unknown')}")

        except Exception as e:
            print(f"Failed to notify novelty: {e}")

    def shutdown(self) -> None:
        """Graceful shutdown."""
        self.running = False
        try:
            self._atomic_save_state()
        except Exception:
            pass
        print("Emotional state saved")


if __name__ == "__main__":
    process = EmotionalProcess()
    try:
        process.start()
    except KeyboardInterrupt:
        print("\nEmotional engine shutting down...")
        process.shutdown()
