# agent.py
import os
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import requests  # Add this import

from deep_translator import GoogleTranslator
from langdetect import detect as lang_detect
from rapidfuzz import process, fuzz

try:
    from matcher import RuleEngine
except Exception:
    RuleEngine = None

DATA_DIR = Path("data")
MODEL_PATH = Path("models/scheme_router_model.joblib") if Path("models").exists() else Path("scheme_router_model.joblib")


class AIAgent:
    def __init__(self):
        self.translator_to_en = GoogleTranslator(source="auto", target="en")

        # Router
        self.router = None
        if MODEL_PATH.exists():
            try:
                self.router = joblib.load(MODEL_PATH)
            except Exception:
                self.router = None

        # Rule engine
        self.rule_engine = RuleEngine() if RuleEngine else None

        # Knowledge base
        self.knowledge_chunks = self._load_kb()

        # API configuration
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip()
            
        self.model = os.getenv("LLM_MODEL")
        if not self.model or not self.model.strip():
            # A very stable free model on OpenRouter
            self.model = "google/gemini-2.0-flash-exp:free"
        else:
            self.model = self.model.strip()
            
        self.base_url = os.getenv("LLM_BASE_URL")
        # Default to a sane value if not provided
        if not self.base_url or not self.base_url.strip():
            self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            self.base_url = self.base_url.strip()
            # If they provided just the base, append the completion path correctly
            if not self.base_url.endswith("/chat/completions"):
                self.base_url = self.base_url.rstrip("/") + "/chat/completions"
        
        print(f"Agent initialized with API key: {bool(self.api_key)}")
        print(f"Using model: {self.model}")
        print(f"Using base URL: {self.base_url}")

    # ---------------- Chatbot Q&A ----------------
    def ask_genai(self, query: str, language: str = "en") -> str:
        """Chat-style answer generator for the floating chatbot widget."""
        if not self.api_key:
            return "❌ API key not configured. Please check your environment variables."
        
        # Detect language if auto
        if language == "auto":
            try:
                code = lang_detect(query)
                if code.startswith("hi"):
                    language = "hi"
                elif code.startswith("te"):
                    language = "te"
                else:
                    language = "en"
            except Exception:
                language = "en"

        # Build system and user messages
        system_msg = (
            "You are an AI assistant for Indian government welfare schemes. "
            "You must answer questions about PM-Kisan, Ration Card, and Pension. "
            "Be accurate, concise (<150 words), and strictly use the requested language. "
            "If user asks outside these schemes, politely say you only support those schemes in the requested language."
        )
        lang_name = "Telugu" if language == "te" else "Hindi" if language == "hi" else "English"
        user_msg = f"CRITICAL: You must answer ONLY in {lang_name}. Citizen query: {query}"

        # Call LLM using direct HTTP request
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Mounika1384/ai-grievance-redressal", # Optional
                    "X-Title": "AI Grievance Redressal Agent" # Optional
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                },
                timeout=15 # Add timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"API Error DEBUG: Status={response.status_code}, URL={self.base_url}, Model={self.model}")
                return f"⚠️ AI Error {response.status_code}: Model '{self.model}' not found at {self.base_url}. Please check your Render Environment Variables."
                
        except Exception as e:
            print(f"Request failed: {e}")
            return "⚠️ Sorry, AI service failed. Please try again."

    # ---------------- Main Query Processing ----------------
    def process_query(self, query_text: str, profile: Dict, language: str = "en") -> Dict:
        target_lang = self._decide_language(language, query_text)
        try:
            query_en = self.translator_to_en.translate(query_text) if target_lang != "en" else query_text
        except Exception:
            query_en = query_text

        scheme = self._route_scheme(query_en, query_text)
        eligibility, reasons = self._check_eligibility(scheme, profile)
        context_passages = self._retrieve_context(scheme, query_en, top_k=5)

        description, documents, next_steps, helplines = self._compose_with_llm(
            query_en=query_en,
            scheme=scheme,
            eligibility=eligibility,
            reasons=reasons,
            profile=profile,
            context_passages=context_passages,
            out_lang=target_lang,
        )

        return {
            "scheme": scheme,
            "eligibility": eligibility,
            "reasons": reasons,
            "description": description,
            "documents": documents,
            "next_steps": next_steps,
            "helplines": helplines,
        }

    # ---------------- Internals ----------------
    def _decide_language(self, language: str, text: str) -> str:
        """Return 'en' | 'hi' | 'te' based on selector or auto-detect."""
        if language and language != "auto":
            return language
        try:
            code = lang_detect(text)
            if code.startswith("hi"):
                return "hi"
            if code.startswith("te"):
                return "te"
            return "en"
        except Exception:
            return "en"

    def _route_scheme(self, query_en: str, original_text: str = "") -> str:
        q = query_en.lower()
        orig = original_text.lower()

        # --- PM-Kisan ---
        if any(k in q for k in ["pm-kisan", "pm kisan", "kisan", "farmer", "ekyc", "installment"]) or \
           any(k in orig for k in ["किसान", "pm किसान", "రైతు", "పిఎం-కిసాన్"]):
            return "PM-KISAN"

        # --- Ration Card ---
        if any(k in q for k in ["ration", "pds", "bpl", "aay", "food security", "ration card"]) or \
           any(k in orig for k in ["राशन", "pds", "రేషన్ కార్డు"]):
            return "RATION CARD"

        # --- Pension ---
        if any(k in q for k in ["pension", "old age", "widow", "disability", "nsap"]) or \
           any(k in orig for k in ["पेंशन", "वृद्धावस्था", "వృద్ధాప్య పెన్షన్", "పెన్షన్"]):
            return "PENSION"

        return "GENERAL"

    def _check_eligibility(self, scheme: str, profile: Dict) -> Tuple[bool, List[str]]:
        """Check eligibility using rules or fallback logic."""
        if self.rule_engine:
            try:
                result = self.rule_engine.check_eligibility(scheme, profile)
                return bool(result.get("eligible", False)), list(result.get("reasons", []))
            except Exception:
                pass

        # --- Fallback ---
        age = int(profile.get("age", 0) or 0)
        income = int(float(profile.get("income", 0) or 0))
        land = float(profile.get("landholding", 0) or 0.0)
        occ = str(profile.get("occupation", "")).lower()

        if scheme == "PM-KISAN":
            ok = (land > 0) and ("farm" in occ or "farmer" in occ)
            reasons = []
            if not (land > 0): reasons.append("Landholding required (>0).")
            if "farm" not in occ and "farmer" not in occ: reasons.append("Occupation should be farmer.")
            return ok, reasons

        if scheme == "RATION CARD":
            ok = income <= 120000
            reasons = [] if ok else [f"Annual income ₹{income} exceeds threshold."]
            return ok, reasons

        if scheme == "PENSION":
            ok = age >= 60
            reasons = [] if ok else [f"Age {age} < 60."]
            return ok, reasons

        return False, ["Could not determine scheme-specific rules."]

    def _load_kb(self) -> List[str]:
        """Load text from txt/csv into a list of chunks."""
        chunks = []

        def _add(text: str):
            if text and isinstance(text, str):
                for piece in self._chunk(text, 600):
                    chunks.append(piece.strip())

        # pmkisan.txt
        for p in [Path("data/pmkisan.txt"), Path("pmkisan.txt")]:
            if p.exists():
                try:
                    _add(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass

        # CSVs
        for name in ["bhagi1.csv", "bhagi2.csv"]:
            for p in [Path(f"data/{name}"), Path(name)]:
                if p.exists():
                    try:
                        df = pd.read_csv(p)
                        for col in df.columns:
                            if df[col].dtype == object:
                                joined = " ".join(map(str, df[col].dropna().tolist()))
                                _add(joined)
                    except Exception:
                        pass

        if not chunks:
            chunks = [
                "PM-Kisan: eKYC needed; small/marginal farmers eligible; Aadhaar & land records required.",
                "Ration Card: income & residence based; Aadhaar, income certificate, residence proof needed.",
                "Pension: age >= 60; Aadhaar, age proof, bank details required; apply via CSC or state portal."
            ]
        return chunks

    def _chunk(self, text: str, max_len: int) -> List[str]:
        words = text.split()
        out, cur = [], []
        for w in words:
            cur.append(w)
            if sum(len(x) + 1 for x in cur) > max_len:
                out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
        return out

    def _retrieve_context(self, scheme: str, query_en: str, top_k: int = 5) -> List[str]:
        if scheme and scheme != "GENERAL":
            scheme_keywords = scheme.lower().split()
            candidates = [c for c in self.knowledge_chunks if any(k in c.lower() for k in scheme_keywords)]
            base = candidates if candidates else self.knowledge_chunks
        else:
            base = self.knowledge_chunks

        results = process.extract(query_en, base, scorer=fuzz.WRatio, limit=top_k)
        return [r[0] for r in results]

    def _compose_with_llm(
        self,
        query_en: str,
        scheme: str,
        eligibility: bool,
        reasons: List[str],
        profile: Dict,
        context_passages: List[str],
        out_lang: str,
    ) -> Tuple[str, List[str], List[str], List[str]]:
        default_docs = {
            "PM-KISAN": ["Aadhaar", "Land records (RoR)", "Bank passbook", "Mobile number"],
            "RATION CARD": ["Aadhaar", "Income certificate", "Residence proof", "Family details"],
            "PENSION": ["Aadhaar", "Age proof", "Bank passbook", "Residence proof"],
            "GENERAL": ["Aadhaar", "Residence proof"]
        }
        default_steps = {
            "PM-KISAN": [
                "Complete e-KYC on PM-Kisan portal or at CSC.",
                "Verify land records and Aadhaar-bank seeding.",
                "Track installment status on the portal."
            ],
            "RATION CARD": [
                "Apply at state PDS portal or CSC.",
                "Upload documents and family details.",
                "Track application using receipt number."
            ],
            "PENSION": [
                "Apply at the state Social Welfare/NSAP portal or CSC.",
                "Submit age and income proof with bank details.",
                "Track approval and first credit in your bank."
            ],
            "GENERAL": ["Visit your nearest CSC for personalized guidance."]
        }
        default_help = ["1800-180-1551 (PM-Kisan)", "State PDS Helpline", "Social Welfare Dept Helpline"]

        system_msg = (
            "You are an AI Grievance Redressal Agent for Indian government schemes. "
            "Be factual, concise (<180 words), and step-by-step. "
            "CRITICAL: You MUST output the entire response in the requested language (e.g., if Telugu is requested, use Telugu script). "
            "If not eligible, explain 1–2 reasons and suggest alternatives."
        )
        lang_name = "Telugu" if out_lang == "te" else "Hindi" if out_lang == "hi" else "English"
        user_payload = {
            "query": query_en,
            "scheme": scheme,
            "eligibility": "Eligible" if eligibility else "Not Eligible",
            "reasons": reasons,
            "profile": profile,
            "language": lang_name,
            "context_passages": context_passages[:5],
            "instructions": [
                "Return a short explanation of eligibility.",
                "Give a numbered checklist to apply.",
                "Mention key documents.",
                "Include 1–2 helplines or official portals if known."
            ]
        }
        user_msg = json.dumps(user_payload, ensure_ascii=False)

        # Use direct HTTP request instead of OpenAI client
        llm_text = None
        if self.api_key:
            try:
                response = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        "temperature": 0.2,
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    llm_text = result['choices'][0]['message']['content'].strip()
                else:
                    print(f"LLM API Error: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"LLM Request failed: {e}")

        if not llm_text:
            base = f"{scheme} — {'Eligible' if eligibility else 'Not Eligible'}."
            if reasons:
                base += " " + " ".join(reasons)
            steps = default_steps.get(scheme, default_steps["GENERAL"])
            docs = default_docs.get(scheme, default_docs["GENERAL"])
            desc = base + " Follow the steps to proceed."
            if out_lang != "en":
                try:
                    desc = GoogleTranslator(source="auto", target=out_lang).translate(desc)
                    steps = [GoogleTranslator(source="auto", target=out_lang).translate(s) for s in steps]
                    docs = [GoogleTranslator(source="auto", target=out_lang).translate(d) for d in docs]
                    help_lines = [GoogleTranslator(source="auto", target=out_lang).translate(h) for h in default_help]
                except Exception:
                    help_lines = default_help
            else:
                help_lines = default_help
            return desc, docs, steps, help_lines

        docs = default_docs.get(scheme, default_docs["GENERAL"])
        steps = default_steps.get(scheme, default_steps["GENERAL"])
        help_lines = default_help

        # We don't translate llm_text here because the LLM was already told to output in out_lang
        if out_lang != "en":
            try:
                translator = GoogleTranslator(source="auto", target=out_lang)
                docs = [translator.translate(d) for d in docs]
                steps = [translator.translate(s) for s in steps]
                help_lines = [translator.translate(h) for h in help_lines]
            except Exception as e:
                print(f"Fallback translation error: {e}")

        return llm_text, docs, steps, help_lines