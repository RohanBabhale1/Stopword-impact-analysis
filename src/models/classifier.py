from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import joblib

class TextClassifier:
    """Wrapper for various classification algorithms"""
    
    def __init__(self, model_type='nb'):
        self.model_type = model_type
        self.model = self._initialize_model(model_type)
    
    def _initialize_model(self, model_type):
        """Initialize classifier based on type"""
        if model_type == 'nb':
            return MultinomialNB()
        elif model_type == 'lr':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            return LinearSVC(random_state=42)
        elif model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """Train the classifier"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)
        else:
            return None
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)