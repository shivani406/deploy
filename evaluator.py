import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class RobustJobMismatchEvaluator:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))

        # Add custom job-related stopwords
        self.job_stop_words = {
            'job', 'position', 'role', 'opportunity', 'candidate', 'employee',
            'work', 'company', 'team', 'looking', 'seeking', 'required',
            'experience', 'skills', 'ability', 'must', 'should', 'will', 'can'
        }
        self.stop_words.update(self.job_stop_words)

    def semantic_similarity_score(self, job_titles, job_descriptions):
        """Compute cosine similarity between job title and description"""
        title_embeddings = self.sentence_model.encode(job_titles)
        desc_embeddings = self.sentence_model.encode(job_descriptions)

        similarities = []
        for i in range(len(job_titles)):
            sim = cosine_similarity([title_embeddings[i]], [desc_embeddings[i]])[0][0]
            similarities.append(max(0, sim))  # Clamp to non-negative

        mismatch_scores = [1 - sim for sim in similarities]

        return {
            'individual_scores': mismatch_scores,
            'average_mismatch': np.mean(mismatch_scores),
            'std_mismatch': np.std(mismatch_scores)
        }

    def extract_meaningful_terms(self, text, min_freq=1, max_terms=15):
        """Extract frequent terms from job text"""
        text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        tokens = word_tokenize(text_clean)

        meaningful_tokens = [
            token for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        term_freq = Counter(meaningful_tokens)
        return [term for term, freq in term_freq.most_common(max_terms) if freq >= min_freq]

    def tfidf_keyword_overlap(self, job_titles, job_descriptions):
        """Check overlap of TF-IDF extracted keywords between title and description"""
        mismatch_scores = []

        for title, desc in zip(job_titles, job_descriptions):
            combined = [title, desc]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            tfidf = vectorizer.fit_transform(combined)

            title_terms = set(vectorizer.inverse_transform(tfidf[0])[0])
            desc_terms = set(vectorizer.inverse_transform(tfidf[1])[0])

            overlap = len(title_terms & desc_terms)
            total = len(title_terms | desc_terms)

            score = 1.0 - (overlap / total) if total > 0 else 1.0
            mismatch_scores.append(score)

        return {
            'individual_scores': mismatch_scores,
            'average_mismatch': np.mean(mismatch_scores),
            'std_mismatch': np.std(mismatch_scores)
        }

    def evaluate(self, job_titles, job_descriptions):
        """
        Evaluate mismatch based on semantic similarity and keyword overlap.
        Expects lists of job titles and job descriptions.
        Returns a combined mismatch score as a percentage (0-100).
        """
        # Semantic similarity
        sim_scores = self.semantic_similarity_score(job_titles, job_descriptions)
        avg_sim_mismatch = sim_scores['average_mismatch']

        # TF-IDF keyword overlap
        tfidf_scores = self.tfidf_keyword_overlap(job_titles, job_descriptions)
        avg_overlap_mismatch = tfidf_scores['average_mismatch']

        # Weighted average of mismatch scores (adjust weights here if needed)
        final_mismatch_score = (0.6 * avg_sim_mismatch) + (0.4 * avg_overlap_mismatch)

        return final_mismatch_score * 100  # Return percentage
