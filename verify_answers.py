"""
verify_answers.py

Checks that DB answers are being loaded correctly, that roles are recognized,
and reports any missing/placeholder ideal_answer entries.
"""
import json
from models.enhanced_interview_model import EnhancedInterviewModel

DB_PATH = "database.jsonl"

def verify_database_answers(db_path=DB_PATH, sample_limit=20):
    model = EnhancedInterviewModel(db_path=db_path)

    print("🔍 VERIFYING DATABASE ANSWERS")
    print("=" * 80)
    print(f"Roles found: {model.get_roles()}")
    print(f"Difficulties found: {model.get_difficulties()}")
    print("-" * 80)

    if not model.questions:
        print("No questions loaded.")
        return

    # sample some questions (first N)
    for i, question in enumerate(model.questions[:sample_limit]):
        print(f"\n{i+1}. ROLE: {question.get('role','Any')} | DIFFICULTY: {question.get('difficulty','Any')}")
        print(f"   QUESTION: {question['question']}")
        stored_answer = question.get('ideal_answer', '')
        if not stored_answer or stored_answer.strip() == "":
            print("   STORED: <MISSING IDEAL ANSWER>")
        else:
            print(f"   STORED: {stored_answer[:150]}{'...' if len(stored_answer) > 150 else ''}")

        fetched = model.get_actual_correct_answer(question)
        print(f"   FETCHED: {fetched[:150]}{'...' if len(fetched) > 150 else ''}")

        # report generic / missing
        is_generic = (not fetched.strip())
        print(f"   IS GENERIC/MISSING: {is_generic}")

if __name__ == "__main__":
    verify_database_answers()
