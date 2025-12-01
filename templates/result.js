// templates/result.js
class ResultTemplate {
    constructor(sessionData) {
        this.sessionData = sessionData;
        this.initializeTemplate();
    }

    initializeTemplate() {
        this.updateSummaryCards();
        this.updateQuestionsAnalysis();
        this.updateCharts();
        this.updateRecommendations();
        this.updateSessionInfo();
    }

    updateSummaryCards() {
        const stats = this.calculateStats();
        
        // Update overall score
        document.getElementById('overall-score').textContent = `${stats.overallScore}/10`;
        document.getElementById('performance-level').textContent = stats.performanceLevel;
        document.getElementById('performance-level').className = `performance-level level-${stats.performanceLevel.toLowerCase()}`;
        
        // Update other cards
        document.getElementById('total-questions').textContent = stats.totalQuestions;
        document.getElementById('session-duration').textContent = stats.sessionDuration;
        document.getElementById('avg-response').textContent = stats.avgResponseTime;
    }

    updateQuestionsAnalysis() {
        const questionsContainer = document.querySelector('.questions-section');
        questionsContainer.innerHTML = '<h2 class="section-title">Detailed Question Analysis</h2>';
        
        this.sessionData.questions.forEach((question, index) => {
            const questionElement = this.createQuestionElement(question, index + 1);
            questionsContainer.appendChild(questionElement);
        });
    }

    createQuestionElement(questionData, questionNumber) {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'question-item';
        
        const evaluation = questionData.evaluation;
        const score = evaluation ? evaluation.score : 'N/A';
        
        questionDiv.innerHTML = `
            <div class="question-header" onclick="toggleQuestion('question${questionNumber}')">
                <div class="question-number">Q${questionNumber}</div>
                <div class="question-text">${this.escapeHtml(questionData.question)}</div>
                <div class="question-score">${score}/10</div>
            </div>
            <div class="question-details" id="question${questionNumber}">
                <div class="answer-section">
                    <h4 class="section-subtitle">Your Answer</h4>
                    <div class="user-answer">
                        ${this.escapeHtml(questionData.user_answer || 'No answer provided')}
                    </div>
                </div>
                ${evaluation ? this.createEvaluationSection(evaluation) : ''}
            </div>
        `;
        
        return questionDiv;
    }

    createEvaluationSection(evaluation) {
        return `
            <div class="evaluation-section">
                <h4 class="section-subtitle">Evaluation Metrics</h4>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value">${evaluation.keyword_score || 'N/A'}</div>
                        <div class="metric-label">Keyword Score</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${evaluation.sentiment_score || 'N/A'}</div>
                        <div class="metric-label">Sentiment Score</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${evaluation.completeness_score || 'N/A'}</div>
                        <div class="metric-label">Completeness</div>
                    </div>
                </div>
                
                <div class="keywords-section">
                    <div class="keywords-positive">
                        <div class="keywords-title">Keywords Matched (${evaluation.matched_keywords ? evaluation.matched_keywords.length : 0})</div>
                        <div class="keyword-list">
                            ${evaluation.matched_keywords ? evaluation.matched_keywords.map(kw => 
                                `<span class="keyword">${this.escapeHtml(kw)}</span>`
                            ).join('') : ''}
                        </div>
                    </div>
                    <div class="keywords-negative">
                        <div class="keywords-title">Keywords Missing (${evaluation.missing_keywords ? evaluation.missing_keywords.length : 0})</div>
                        <div class="keyword-list">
                            ${evaluation.missing_keywords ? evaluation.missing_keywords.map(kw => 
                                `<span class="keyword">${this.escapeHtml(kw)}</span>`
                            ).join('') : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    updateCharts() {
        // This would integrate with Chart.js or similar library
        // For now, it's placeholder content
        console.log('Charts would be generated here with data:', this.sessionData);
    }

    updateRecommendations() {
        const recommendations = this.generateRecommendations();
        const container = document.querySelector('.recommendations');
        
        let recommendationsHTML = '<h2 class="section-title">Improvement Recommendations</h2>';
        
        recommendations.forEach(rec => {
            recommendationsHTML += `
                <div class="recommendation-item">
                    <div class="recommendation-icon">${rec.icon}</div>
                    <div>
                        <strong>${rec.title}:</strong> ${rec.description}
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = recommendationsHTML;
    }

    updateSessionInfo() {
        document.getElementById('session-date').textContent = 
            new Date().toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            });
    }

    calculateStats() {
        const questions = this.sessionData.questions;
        const totalQuestions = questions.length;
        
        // Calculate overall score
        const totalScore = questions.reduce((sum, q) => 
            sum + (q.evaluation ? q.evaluation.score : 0), 0);
        const overallScore = totalQuestions > 0 ? (totalScore / totalQuestions).toFixed(1) : 0;
        
        // Determine performance level
        let performanceLevel = 'Excellent';
        if (overallScore < 6) performanceLevel = 'Poor';
        else if (overallScore < 7) performanceLevel = 'Average';
        else if (overallScore < 8) performanceLevel = 'Good';
        
        return {
            overallScore,
            performanceLevel,
            totalQuestions,
            sessionDuration: this.formatDuration(this.sessionData.duration),
            avgResponseTime: this.calculateAvgResponseTime()
        };
    }

    generateRecommendations() {
        const recommendations = [];
        const stats = this.calculateStats();
        
        if (stats.overallScore < 7) {
            recommendations.push({
                icon: '🎯',
                title: 'Focus on Content',
                description: 'Work on providing more detailed and structured answers with specific examples.'
            });
        }
        
        if (stats.overallScore < 8) {
            recommendations.push({
                icon: '💡',
                title: 'Improve Keyword Usage',
                description: 'Ensure you address all key points mentioned in the question for better scoring.'
            });
        }
        
        recommendations.push({
            icon: '⚡',
            title: 'Practice Regularly',
            description: 'Continue practicing with different question types to improve your response time and quality.'
        });
        
        return recommendations;
    }

    formatDuration(seconds) {
        if (!seconds) return 'N/A';
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    calculateAvgResponseTime() {
        // This would be calculated based on actual response times
        return '45s';
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
}

// Global function to toggle questions
function toggleQuestion(questionId) {
    const element = document.getElementById(questionId);
    if (element) {
        element.classList.toggle('active');
    }
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    // This would be populated with actual session data
    const sessionData = window.sessionData || {};
    window.resultTemplate = new ResultTemplate(sessionData);
});