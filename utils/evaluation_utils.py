import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
import datetime
import tempfile
import os

class EvaluationReport:
    def __init__(self, session_data):
        self.session_data = session_data
    
    def calculate_overall_stats(self):
        questions = self.session_data.get('questions', [])
        if not questions:
            return self.get_empty_stats()
        
        scores = [q.get('evaluation', {}).get('score', 0) for q in questions]
        categories = {}
        
        for q in questions:
            cat = q.get('category', 'general')
            if cat not in categories:
                categories[cat] = []
            eval_data = q.get('evaluation', {})
            categories[cat].append(eval_data.get('score', 0))
        
        category_scores = {cat: np.mean(scores) if scores else 0 for cat, scores in categories.items()}
        
        return {
            'overall_score': np.mean(scores) if scores else 0,
            'total_questions': len(scores),
            'category_scores': category_scores,
            'performance_level': self.get_performance_level(np.mean(scores) if scores else 0),
            'duration': self.session_data.get('end_time', 0) - self.session_data.get('start_time', 0)
        }
    
    def get_empty_stats(self):
        return {
            'overall_score': 0,
            'total_questions': 0,
            'category_scores': {},
            'performance_level': "No Data",
            'duration': 0
        }
    
    def get_performance_level(self, score):
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Average"
        else:
            return "Needs Improvement"
    
    def generate_report_text(self):
        stats = self.calculate_overall_stats()
        
        report = f"""
INTERVIEW PERFORMANCE REPORT
============================

Session ID: {self.session_data.get('session_id', 'N/A')}
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Score: {stats['overall_score']:.2f}/10
Performance Level: {stats['performance_level']}
Total Questions: {stats['total_questions']}
Session Duration: {self.format_duration(stats['duration'])}

Category-wise Performance:
"""
        
        for category, score in stats['category_scores'].items():
            report += f"\n{category.title()}: {score:.2f}/10"
        
        report += "\n\nDetailed Analysis:\n"
        
        for i, q_data in enumerate(self.session_data.get('questions', []), 1):
            eval_data = q_data.get('evaluation', {})
            report += f"\n{i}. {q_data.get('question', 'N/A')}"
            report += f"\n   Your Answer: {q_data.get('user_answer', 'No answer provided')}"
            report += f"\n   Score: {eval_data.get('score', 'N/A')}/10"
            report += f"\n   Keywords Matched: {len(eval_data.get('matched_keywords', []))}/{len(eval_data.get('matched_keywords', []) + eval_data.get('missing_keywords', []))}"
            
            matched = eval_data.get('matched_keywords', [])
            if matched:
                report += f"\n   Matched Keywords: {', '.join(matched)}"
            
            missing = eval_data.get('missing_keywords', [])
            if missing:
                report += f"\n   Missing Keywords: {', '.join(missing)}"
            
            # Add ideal answer to report - get from evaluation data first, then question data
            ideal_answer = eval_data.get('ideal_answer', q_data.get('ideal_answer', ''))
            if ideal_answer:
                report += f"\n   Model Answer: {ideal_answer}"
            
            report += "\n"
        
        return report
    
    def format_duration(self, seconds):
        """Format duration in seconds to readable string"""
        if not seconds:
            return "N/A"
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"
    
    def generate_pdf_report(self, filename):
        """Generate PDF report with proper file handling"""
        try:
            # Create a temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                temp_filename = tmp_file.name
            
            doc = SimpleDocTemplate(temp_filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("Interview Performance Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Session info
            session_info = f"""
            Session ID: {self.session_data.get('session_id', 'N/A')}<br/>
            Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            """
            story.append(Paragraph(session_info, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Overall stats
            stats = self.calculate_overall_stats()
            overall_text = f"""
            Overall Score: {stats['overall_score']:.2f}/10<br/>
            Performance Level: {stats['performance_level']}<br/>
            Total Questions: {stats['total_questions']}<br/>
            Duration: {self.format_duration(stats['duration'])}
            """
            story.append(Paragraph(overall_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Questions analysis
            story.append(Paragraph("Detailed Question Analysis", styles['Heading2']))
            
            for i, q_data in enumerate(self.session_data.get('questions', []), 1):
                eval_data = q_data.get('evaluation', {})
                
                q_text = Paragraph(f"<b>Q{i}: {q_data.get('question', 'N/A')}</b>", styles['Normal'])
                story.append(q_text)
                
                ans_text = Paragraph(f"<b>Your Answer:</b> {q_data.get('user_answer', 'No answer')}", styles['Normal'])
                story.append(ans_text)
                
                score_text = Paragraph(f"<b>Score:</b> {eval_data.get('score', 'N/A')}/10", styles['Normal'])
                story.append(score_text)
                
                # Add ideal answer - get from evaluation data first, then question data
                ideal_answer = eval_data.get('ideal_answer', q_data.get('ideal_answer', ''))
                if ideal_answer:
                    ideal_text = Paragraph(f"<b>Model Answer:</b> {ideal_answer}", styles['Normal'])
                    story.append(ideal_text)
                
                story.append(Spacer(1, 12))
            
            doc.build(story)
            
            # Move the temporary file to the final destination
            import shutil
            shutil.move(temp_filename, filename)
            return True
            
        except Exception as e:
            print(f"PDF generation failed: {e}")
            # Clean up temporary file if it exists
            try:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except:
                pass
            return False