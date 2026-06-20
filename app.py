from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

# ─────────────────────────────────────────────
# MEDICAL KNOWLEDGE BASE
# ─────────────────────────────────────────────
MEDICAL_DATA = [
    "ACNE: A skin condition caused by clogged hair follicles. Symptoms: pimples, blackheads, whiteheads on face, chest, back. Causes: excess oil, bacteria, hormonal changes. Treatment: benzoyl peroxide, salicylic acid, topical antibiotics. See doctor for severe cases.",
    "DIABETES TYPE 1: Autoimmune condition where pancreas makes no insulin. Symptoms: frequent urination, extreme thirst, weight loss, fatigue, blurred vision. Treatment: insulin injections, blood sugar monitoring, healthy diet.",
    "DIABETES TYPE 2: Body resists insulin. Symptoms: frequent urination, thirst, fatigue, slow healing wounds. Treatment: Metformin, lifestyle changes, diet, exercise. Risk factors: obesity, age over 45, family history.",
    "FEVER: Body temperature above 38 degrees C or 100.4 F. Causes: viral or bacterial infection. Treatment: paracetamol 500-1000mg, ibuprofen, rest, drink plenty of fluids. See doctor if above 39.5C, lasts more than 3 days, or with breathing difficulty.",
    "HEADACHE TENSION: Band-like pressure around head. Treatment: paracetamol or ibuprofen, rest, reduce stress.",
    "HEADACHE MIGRAINE: One-sided pulsating pain with nausea, vomiting, sensitivity to light and sound. Lasts 4-72 hours. Treatment: triptans, pain relievers, rest in dark quiet room. Triggers: stress, hormones, certain foods.",
    "HEADACHE EMERGENCY: Sudden severe thunderclap headache, headache with stiff neck and fever, headache after head injury - go to emergency immediately.",
    "HYPERTENSION HIGH BLOOD PRESSURE: Blood pressure above 130/80 mmHg. Usually no symptoms. Severe cases: headache, nosebleed. Treatment: lifestyle changes, ACE inhibitors like lisinopril, beta blockers like metoprolol, amlodipine. Complications: heart attack, stroke.",
    "ASTHMA: Chronic airway inflammation. Symptoms: wheezing, shortness of breath, chest tightness, night cough. Triggers: smoke, allergens, cold air, exercise. Treatment: inhaled corticosteroids as preventer, bronchodilators as reliever.",
    "COMMON COLD: Viral infection of upper respiratory tract. Symptoms: runny nose, sore throat, cough, congestion, mild fever. Treatment: rest, fluids, paracetamol, nasal decongestants. Resolves in 7-10 days.",
    "CHEST PAIN EMERGENCY: Central crushing chest pain with sweating and nausea radiating to arm or jaw - call 112 immediately. Could be heart attack.",
    "APPENDICITIS: Severe pain starting around belly button moving to lower right abdomen. Nausea, vomiting, fever. EMERGENCY - go to hospital immediately.",
    "PARACETAMOL: 500-1000mg every 4-6 hours. Maximum 4 grams per day. Do not take with alcohol. Safe for most people including pregnant women.",
    "IBUPROFEN: 200-400mg every 4-6 hours. Take with food. Avoid in kidney disease, peptic ulcer, pregnancy third trimester, heart disease.",
    "METFORMIN: First line treatment for Type 2 diabetes. 500-2000mg daily with meals. Side effect: nausea and stomach upset. Do not use in severe kidney disease.",
    "SKIN RASH: Could indicate allergy, eczema, psoriasis, or infection. Eczema: itchy inflamed skin. Psoriasis: red patches with silver scales. Allergic rash: hives with itching. See doctor for diagnosis.",
    "STOMACH PAIN: Upper right - gallbladder or liver. Upper left - stomach or pancreas. Lower right - appendicitis emergency. Lower left - IBS or diverticulitis. Central - stomach ulcer or IBS.",
    "SORE THROAT: Viral: gradual onset with cold symptoms. Bacterial strep: sudden severe sore throat with white patches and fever needs antibiotics. Gargle warm salt water, drink warm fluids, take paracetamol.",
    "DIARRHEA: Loose watery stools. Causes: viral gastroenteritis, food poisoning, IBS. Treatment: oral rehydration salts, bland diet, avoid dairy. See doctor if blood in stool, lasts more than 3 days, or severe dehydration.",
    "URINARY TRACT INFECTION UTI: Burning during urination, frequent urination, cloudy urine, pelvic pain. More common in women. Treatment: antibiotics prescribed by doctor, drink plenty of water.",
]

# ─────────────────────────────────────────────
# INITIALIZE MODELS (Lazy loading)
# ─────────────────────────────────────────────
embedding_model = None
index = None
client = None
chat_history = {}

def initialize_models(api_key):
    global embedding_model, index, client
    
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(MEDICAL_DATA)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        print("✓ Embedding model loaded")
    
    if client is None:
        print("Connecting to Groq...")
        try:
            client = Groq(api_key=api_key)
        except TypeError as e:
            # Handle proxies argument if present in environment
            import os
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)
            client = Groq(api_key=api_key)
        print("✓ Groq connected")

def get_context(question):
    q_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(q_embedding, dtype=np.float32), k=3)
    return "\n".join([MEDICAL_DATA[i] for i in indices[0]])

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('medical_ui.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        api_key = data.get('apiKey')
        question = data.get('question', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Initialize models with API key
        try:
            initialize_models(api_key)
        except Exception as e:
            return jsonify({'error': f'Failed to initialize: {str(e)}'}), 500
        
        # Emergency check
        emergency_words = ["chest pain", "heart attack", "stroke", "not breathing",
                          "unconscious", "overdose", "severe bleeding"]
        if any(word in question.lower() for word in emergency_words):
            return jsonify({
                'answer': '🚨 EMERGENCY ALERT!\n\nPlease call 112 or 108 immediately!\n\nDo NOT rely on AI for medical emergencies. Get professional help right away.',
                'is_emergency': True
            })
        
        # Initialize session history if needed
        if session_id not in chat_history:
            chat_history[session_id] = []
        
        # Get relevant medical context
        context = get_context(question)
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": f"""You are MediBot, a helpful and empathetic medical AI assistant.
Use the medical knowledge below to answer the patient's question clearly.
Give detailed, helpful answers. Always recommend consulting a real doctor for diagnosis or treatment.
Never diagnose. Always be empathetic and supportive.

MEDICAL KNOWLEDGE:
{context}

IMPORTANT: Always end your response with a reminder to consult a doctor if needed."""
            }
        ]
        
        # Add last 6 messages of history
        for msg in chat_history[session_id][-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": question})
        
        # Get response from Groq
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_tokens=512
        )
        
        answer = response.choices[0].message.content
        
        # Save to history
        chat_history[session_id].append({"role": "user", "content": question})
        chat_history[session_id].append({"role": "assistant", "content": answer})
        
        return jsonify({
            'answer': answer,
            'is_emergency': False
        })
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/validate-key', methods=['POST'])
def validate_key():
    try:
        data = request.get_json()
        api_key = data.get('apiKey', '').strip()
        
        if not api_key or not api_key.startswith('gsk_'):
            return jsonify({'valid': False, 'message': 'Invalid API key format'}), 400
        
        # Try to initialize with the key to validate it
        try:
            # Clear proxy environment variables
            import os
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)
            
            test_client = Groq(api_key=api_key)
            # Make a simple test call
            test_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        except Exception as e:
            return jsonify({'valid': False, 'message': f'Invalid API key: {str(e)}'}), 401
        
        return jsonify({'valid': True, 'message': 'API key is valid'})
    
    except Exception as e:
        return jsonify({'valid': False, 'message': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("  MEDICAL AI CHATBOT - FLASK SERVER")
    print("=" * 50)
    print("\n🚀 Starting server on http://localhost:5000")
    print("   Open your browser and go to http://localhost:5000")
    print("\n" + "=" * 50 + "\n")
    
    app.run(debug=True, host='localhost', port=5000)
