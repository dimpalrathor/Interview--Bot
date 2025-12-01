import json
import random
import logging
import os

logger = logging.getLogger(__name__)

class EnhancedInterviewModel:
    """
    Loads database.jsonl and provides:
      - get_questions(role, difficulty, limit)
      - evaluate_answer(question_dict, user_answer)
    """

    def __init__(self, db_path="database.jsonl"):
        self.db_path = db_path
        self.db = []          # <-- FIXED (was missing)

        if not os.path.exists(self.db_path):
            logger.error("Database file not found: %s", self.db_path)
            return

        self._load_db()

    def _load_db(self):
        """Load JSONL database into memory"""
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # skip empty or comments
                    if not line or line.startswith("#"):
                        continue
                    try:
                        item = json.loads(line)
                        self.db.append(item)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON skipped: %s", line)
        except Exception as e:
            logger.exception("Failed loading DB: %s", e)

        logger.info("Loaded %d questions from DB.", len(self.db))

    # ---------------------------------------------------------------------

    def get_questions(self, role, difficulty, limit=5):
        """
        Return list of question dicts filtered by:
            - role
            - difficulty
        """

        role = role.lower().strip()
        difficulty = difficulty.lower().strip()

        logger.info("Filtering DB: role=%s difficulty=%s", role, difficulty)

        # Filter
        filtered = [
            q for q in self.db
            if q.get("role","").lower() == role
               and q.get("difficulty","").lower() == difficulty
        ]

        logger.info("Found %d matching questions.", len(filtered))

        if not filtered:
            return []

        random.shuffle(filtered)
        return filtered[:limit]

    # ---------------------------------------------------------------------

    def evaluate_answer(self, question_dict, user_answer):
        """
        Very simple scoring: compare keywords.
        """

        correct = question_dict.get("answer", "").lower()
        user = (user_answer or "").lower()

        if not correct or not user:
            question_dict["score"] = 0
            question_dict["correct_answer"] = correct
            return question_dict

        # cheap keyword scoring
        score = 0
        keywords = [w for w in correct.split() if len(w) > 5][:5]

        for kw in keywords:
            if kw in user:
                score += 2

        # clamp 0–10
        score = max(0, min(score, 10))

        question_dict["score"] = score
        question_dict["correct_answer"] = question_dict.get("answer","")
        return question_dict
