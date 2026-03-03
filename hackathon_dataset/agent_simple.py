import pandas as pd
import re

class SimpleAIAgent:
    def __init__(self):
        self.schemes_data = self.load_knowledge_base()
    
    def load_knowledge_base(self):
        # Simple knowledge base without external files
        return {
            'PM-Kisan': {
                'description': 'Pradhan Mantri Kisan Samman Nidhi scheme for farmers',
                'eligibility': 'Farmers with cultivable landholding',
                'income_limit': 150000,
                'land_required': True,
                'min_age': 18,
                'documents': ['Land records', 'Aadhaar card', 'Bank account details']
            },
            'Ration Card': {
                'description': 'Provides subsidized food grains to eligible households',
                'eligibility': 'Based on household income and composition',
                'income_limit': 100000,
                'land_required': False,
                'min_age': 0,
                'documents': ['Address proof', 'Income certificate', 'Family composition proof']
            },
            'Pension': {
                'description': 'Social security pension schemes for elderly, widows, and disabled',
                'eligibility': 'Age 60+ or specific categories',
                'income_limit': 100000,
                'land_required': False,
                'min_age': 60,
                'documents': ['Age proof', 'Income certificate', 'Bank account details']
            }
        }
    
    def detect_scheme(self, query):
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['kisan', 'farmer', 'agriculture', 'land', 'crop']):
            return 'PM-Kisan'
        elif any(word in query_lower for word in ['ration', 'food', 'grains', 'wheat', 'rice']):
            return 'Ration Card'
        elif any(word in query_lower for word in ['pension', 'old age', 'retirement', 'widow', 'disabled']):
            return 'Pension'
        else:
            return 'PM-Kisan'  # Default to PM-Kisan
    
    def extract_profile(self, query):
        profile = {}
        
        # Extract age
        age_match = re.search(r'(\d+)\s*(years|year|yrs|yr|age)', query, re.IGNORECASE)
        if age_match:
            profile['age'] = int(age_match.group(1))
        
        # Extract income
        income_match = re.search(r'(\d+)\s*(rs|inr|₹|rupees|income|salary)', query, re.IGNORECASE)
        if income_match:
            profile['income'] = int(income_match.group(1))
        
        # Extract landholding
        land_match = re.search(r'(\d+)\s*(acres|acre|hectares|hectare|land)', query, re.IGNORECASE)
        if land_match:
            profile['landholding'] = int(land_match.group(1))
        
        return profile
    
    def check_eligibility(self, scheme_name, profile):
        scheme = self.schemes_data[scheme_name]
        reasons = []
        eligible = True
        
        # Check age
        if 'age' in profile and profile['age'] < scheme.get('min_age', 0):
            eligible = False
            reasons.append(f"Minimum age requirement not met (required: {scheme['min_age']}+)")
        
        # Check income
        if 'income' in profile and profile['income'] > scheme.get('income_limit', 0):
            eligible = False
            reasons.append(f"Income exceeds limit (limit: ₹{scheme['income_limit']})")
        
        # Check land requirement
        if scheme.get('land_required', False) and ('landholding' not in profile or profile['landholding'] <= 0):
            eligible = False
            reasons.append("Landholding required but not specified")
        
        return {
            'eligible': eligible,
            'reasons': reasons
        }
    
    def process_query(self, query_text, user_profile, language='en'):
        # Extract profile from query
        extracted_profile = self.extract_profile(query_text)
        profile = {**user_profile, **extracted_profile}
        
        # Detect scheme
        scheme_name = self.detect_scheme(query_text)
        
        # Check eligibility
        eligibility_result = self.check_eligibility(scheme_name, profile)
        
        # Get scheme info
        scheme_info = self.schemes_data[scheme_name]
        
        # Prepare response
        return {
            'scheme': scheme_name,
            'eligibility': eligibility_result['eligible'],
            'reasons': eligibility_result['reasons'],
            'description': scheme_info['description'],
            'documents': scheme_info['documents'],
            'next_steps': self.generate_next_steps(scheme_name, eligibility_result['eligible']),
            'helplines': ['1800-180-1551', '1800-11-0001']
        }
    
    def generate_next_steps(self, scheme_name, is_eligible):
        if not is_eligible:
            return ["Explore alternative schemes", "Contact local authority for clarification"]
        
        next_steps = {
            'PM-Kisan': [
                "Visit https://pmkisan.gov.in",
                "Register with local agriculture office",
                "Submit land documents and bank details"
            ],
            'Ration Card': [
                "Visit state food portal or local ration office",
                "Submit application with required documents",
                "Get biometric verification done"
            ],
            'Pension': [
                "Visit local social welfare office",
                "Submit application with age and income proof",
                "Provide bank account details for direct transfer"
            ]
        }
        
        return next_steps.get(scheme_name, ["Visit official government portal"])