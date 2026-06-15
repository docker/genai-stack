from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai
import anthropic
from langchain_community.llms import Ollama

app = Flask(__name__)
load_dotenv()

# Model configurations
MODEL_CONFIGS = {
    'openai': {
        'name': 'OpenAI',
        'models': [
            'gpt-4-turbo-preview',
            'gpt-4',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k'
        ],
        'api_key_env': 'OPENAI_API_KEY'
    },
    'gemini': {
        'name': 'Google Gemini',
        'models': [
            'gemini-pro',
            'gemini-pro-vision',
            'gemini-2.0-flash'

        ],
        'api_key_env': 'GOOGLE_API_KEY'
    },
    'claude': {
        'name': 'Anthropic Claude',
        'models': [
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307'
        ],
        'api_key_env': 'ANTHROPIC_API_KEY'
    },
    'ollama': {
        'name': 'Ollama',
        'models': [
            'llama2',
            'llama2:13b',
            'llama2:70b',
            'mistral',
            'mistral-openorca',
            'codellama',
            'codellama:13b',
            'codellama:34b',
            'neural-chat',
            'starling-lm',
            'dolphin-phi',
            'orca-mini',
            'vicuna',
            'wizard-vicuna-uncensored'
        ],
        'api_key_env': None
    },
    'cohere': {
        'name': 'Cohere',
        'models': [
            'command',
            'command-light',
            'command-r',
            'command-r-plus'
        ],
        'api_key_env': 'COHERE_API_KEY'
    },
    'huggingface': {
        'name': 'Hugging Face',
        'models': [
            'meta-llama/Llama-2-7b-chat-hf',
            'meta-llama/Llama-2-13b-chat-hf',
            'meta-llama/Llama-2-70b-chat-hf',
            'mistralai/Mistral-7B-Instruct-v0.2',
            'google/flan-t5-xxl',
            'google/flan-ul2'
        ],
        'api_key_env': 'HUGGINGFACE_API_KEY'
    }
}

def get_available_models():
    available_models = []
    for provider, config in MODEL_CONFIGS.items():
        if config['api_key_env'] is None or os.getenv(config['api_key_env']):
            available_models.append({
                'provider': provider,
                'name': config['name'],
                'models': config['models']
            })
    return available_models

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    return jsonify(get_available_models())

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    model = data.get('model')
    provider = data.get('provider')
    temperature = float(data.get('temperature', 0.7))
    max_tokens = int(data.get('max_tokens', 2000))
    
    if not message or not model or not provider:
        return jsonify({'error': 'Missing required parameters'}), 400

    try:
        if provider == 'openai':
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return jsonify({'response': response.choices[0].message.content})
            
        elif provider == 'gemini':
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel(model)
            response = model.generate_content(
                message,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return jsonify({'response': response.text})
            
        elif provider == 'claude':
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": message}]
            )
            return jsonify({'response': response.content[0].text})
            
        elif provider == 'ollama':
            llm = Ollama(
                model=model,
                temperature=temperature,
                num_predict=max_tokens
            )
            response = llm.invoke(message)
            return jsonify({'response': response})
            
        elif provider == 'cohere':
            import cohere
            co = cohere.Client(os.getenv('COHERE_API_KEY'))
            response = co.generate(
                prompt=message,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return jsonify({'response': response.generations[0].text})
            
        elif provider == 'huggingface':
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=os.getenv('HUGGINGFACE_API_KEY'))
            response = client.text_generation(
                message,
                model=model,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            return jsonify({'response': response})
            
        else:
            return jsonify({'error': 'Unsupported provider'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 