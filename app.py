import re
from flask import Flask, request, jsonify, render_template
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

app = Flask(__name__)

kb = {
    "Flu": ["fever", "cough", "sore throat", "fatigue", "headache"],
    "Common Cold": ["cough", "sore throat", "runny nose", "mild fever"],
    "COVID-19": ["fever", "cough", "loss of taste", "loss of smell", "shortness of breath"],
    "Malaria": ["fever", "night sweats", "chills", "fatigue", "vomiting"],
    "Dengue": ["fever", "rash", "joint pain", "headache", "nausea"],
    "Food Poisoning": ["nausea", "vomiting", "diarrhea", "abdominal pain"],
    "Pneumonia": ["fever", "chest pain", "cough", "shortness of breath"],
    "Migraine": ["headache", "nausea", "sensitivity to light", "fatigue"],
    "Tuberculosis": ["cough", "night sweats", "weight loss", "chest pain"],
    "Asthma": ["shortness of breath", "chest tightness", "cough"],
    "Diabetes": ["frequent urination", "excessive thirst", "weight loss", "fatigue", "blurred vision"],
    "Hypertension": ["headache", "dizziness", "chest pain", "shortness of breath", "nosebleeds"],
    "Heart Attack": ["chest pain", "shortness of breath", "sweating", "nausea", "pain in left arm"],
    "Stroke": ["sudden numbness", "confusion", "trouble speaking", "blurred vision", "loss of balance"],
    "Chickenpox": ["fever", "rash", "itching", "loss of appetite", "fatigue"],
    "Typhoid": ["fever", "abdominal pain", "headache", "diarrhea", "loss of appetite"],
    "Hepatitis": ["jaundice", "fatigue", "abdominal pain", "loss of appetite", "dark urine"],
    "Kidney Stones": ["severe back pain", "abdominal pain", "blood in urine", "nausea", "frequent urination"],
    "Anemia": ["fatigue", "shortness of breath", "dizziness", "headache"],
    "Allergy": ["sneezing", "runny nose", "itchy eyes", "rash", "cough"]
}

def extract_symptoms(text):
    tokens = re.findall(r'\w+', text.lower())
    all_symptoms = {s for disease in kb.values() for s in disease}
    return [t for t in tokens if t in all_symptoms]

def rule_based_diagnose(symptoms):
    scores = {}
    for disease, disease_symptoms in kb.items():
        matches = len(set(symptoms) & set(disease_symptoms))
        probability = matches / len(disease_symptoms)
        if matches > 0:
            scores[disease] = round(probability, 2)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

def build_bayesian_model():
    priors = {
        "Flu": 0.10, "Common Cold": 0.15, "COVID-19": 0.05, "Malaria": 0.01,
        "Dengue": 0.01, "Food Poisoning": 0.07, "Pneumonia": 0.03, "Migraine": 0.10,
        "Tuberculosis": 0.01, "Asthma": 0.05, "Diabetes": 0.08, "Hypertension": 0.10,
        "Heart Attack": 0.02, "Stroke": 0.01, "Chickenpox": 0.01, "Typhoid": 0.01,
        "Hepatitis": 0.01, "Kidney Stones": 0.01, "Anemia": 0.05, "Allergy": 0.10
    }

    model = DiscreteBayesianNetwork()
    cpd_list = []

    for disease, prob in priors.items():
        cpd = TabularCPD(
            variable=disease,
            variable_card=2,
            values=[[prob], [1 - prob]],
            state_names={disease: ['yes', 'no']}
        )
        cpd_list.append(cpd)

    symptom_to_diseases = {}
    for disease, symptoms in kb.items():
        for sym in symptoms:
            symptom_to_diseases.setdefault(sym, []).append(disease)

    for sym, diseases in symptom_to_diseases.items():
        for disease in diseases:
            model.add_edge(disease, sym)

        n = len(diseases)
        values = [[], []]

        specificity = max(0.6, 1.0 - (n * 0.1))
        for i in range(2**n):
            parent_state = [(i >> j) & 1 for j in reversed(range(n))]
            if any(parent_state):
                values[0].append(specificity)
                values[1].append(1 - specificity)
            else:
                values[0].append(0.1)
                values[1].append(0.9)

        cpd = TabularCPD(
            variable=sym,
            variable_card=2,
            values=values,
            evidence=diseases,
            evidence_card=[2] * n,
            state_names={sym: ['yes', 'no'], **{d: ['yes', 'no'] for d in diseases}}
        )
        cpd_list.append(cpd)

    model.add_cpds(*cpd_list)
    return model

def bayesian_diagnose(symptoms, model):
    infer = VariableElimination(model)
    evidence_dict = {sym: 'yes' for sym in symptoms}
    results = {}
    for disease in kb.keys():
        try:
            q = infer.query([disease], evidence=evidence_dict, show_progress=False)
            prob_yes = q.values[0]
            results[disease] = round(prob_yes, 2)
        except Exception:
            continue
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

bayesian_model = build_bayesian_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.get_json()
        user_input = data.get('message', '')

        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        symptoms = extract_symptoms(user_input)

        if not symptoms:
            return jsonify({
                'symptoms': [],
                'rule_based': {},
                'bayesian': {},
                'message': 'No symptoms detected'
            })

        rb_results = rule_based_diagnose(symptoms)
        bayes_results = bayesian_diagnose(symptoms, bayesian_model)

        return jsonify({
            'symptoms': symptoms,
            'rule_based': rb_results,
            'bayesian': bayes_results,
            'message': 'Analysis completed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    all_symptoms = sorted(list(set(s for disease in kb.values() for s in disease)))
    return jsonify({'symptoms': all_symptoms})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
