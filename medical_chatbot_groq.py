import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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
# SETUP EMBEDDINGS AND VECTOR SEARCH
# ─────────────────────────────────────────────
print("="*50)
print("  MEDICAL AI CHATBOT")
print("="*50)
print("[1/3] Loading AI embedding model...")
print("      (First time takes 1-2 minutes to download)")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(MEDICAL_DATA)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))
print("[2/3] Knowledge base indexed successfully.")

# ─────────────────────────────────────────────
# GROQ API SETUP
# ─────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    print("[3/3] Enter your FREE Groq API key.")
    print("      Get it at: https://console.groq.com -> API Keys -> Create API Key")
    api_key = input("      Groq API Key: ").strip()

if not api_key:
    print("Error: Groq API key is required. Exiting.")
    exit(1)

client = Groq(api_key=api_key)
chat_history = []
print("[3/3] Groq connected successfully.")

# ─────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("  Chatbot is ready! Ask your medical questions.")
print("  Commands: 'quit' to exit | 'clear' to reset chat")
print("  DISCLAIMER: Educational only. Always see a doctor.")
print("="*50 + "\n")

def get_context(question):
    q_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(q_embedding, dtype=np.float32), k=3)
    return "\n".join([MEDICAL_DATA[i] for i in indices[0]])

while True:
    try:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print("Goodbye! Stay healthy!")
            break

        if question.lower() == "clear":
            chat_history = []
            print("Chat history cleared.\n")
            continue

        # Emergency check
        emergency_words = ["chest pain", "heart attack", "stroke", "not breathing",
                           "unconscious", "overdose", "severe bleeding"]
        if any(word in question.lower() for word in emergency_words):
            print("\nMediBot: EMERGENCY - Please call 112 or 108 immediately!\n")
            continue

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
        for msg in chat_history[-6:]:
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

        print(f"\nMediBot: {answer}\n")

        # Save to history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt:
        print("\nGoodbye! Stay healthy!")
        break
    except Exception as e:
        print(f"\n[Error] {str(e)}\n")
