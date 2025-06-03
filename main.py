from bs4 import BeautifulSoup
import os, dotenv, requests, time, google.generativeai as gemini
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
dotenv.load_dotenv()
gemini.configure(api_key = os.getenv('GEMINI_KEY'))
model = gemini.GenerativeModel("gemini-1.5-flash")

"""
AI functions (backend)
"""

def translate(src, lang):
    prompt = f"""Translate {src} to the language {lang}. It's okay if the requested language to translate to is something funny and not a real language, just try.
    The user wants to be entertained, so be extremely creative and extreme with your translation. Be as creative as possible. However, just give one translation.
    Absolutely, under any circumstance, do NOT give anything else other than just the following:\n
    Prompt: src = Hello, lang = Spanish\n
    Response:\n
    Translation:\nHola"
    Give your response with the exact format given below:\n
    Translation:\n*insert translation here*"""

    response = model.generate_content(prompt)
    response_text = response.text.strip()

    print("[INFO] Unprocessed translation response:", response_text)

    if "Translation:" not in response_text:
        print("[ERR] Failed to translate!!!")
        return "Failed to translate"

    # find translation
    translation = response_text.split(":")[1].strip()
    print("[INFO] Translation:", translation)

    return translation

def mass_translate(text_list, lang, batch_size=500, max_retries=2):
    all_translations = []

    def translate_chunk(chunk, depth=0):
        nonlocal lang, max_retries

        cleaned_chunk = []
        for t in chunk:
            t = t.replace('\n', ' ').replace('\r', '').strip()
            if len(t) > 300:
                t = t[:300] + "..."
            cleaned_chunk.append(t)

        for attempt in range(1, max_retries + 1):
            print(f"[INFO] Translating chunk of size {len(cleaned_chunk)} (Attempt {attempt}/{max_retries})")

            prompt = f"""Translate the following list of text fragments into "{lang}". It's okay if the requested language is funny or fictional. 
            Be extremely creative and entertaining.
            Only return the translations, one per line, each prefixed by >>> and in the same order. No commentary or explanation.
            In this case, you were asked to translate {len(cleaned_chunk)} lines so your output should have {len(cleaned_chunk)} lines exactly.
            The last line you should output should be equivalent to the last number you spit out. Make sure every number prefixing the >>> is three digits,
            if it is not a three digit number, add the appropriate amount of zeroes. Ex: 50 becomes 050.

            Example:
            1. Hello
            2. Goodbye

            Response:
            001 >>> Hola
            002 >>> Adiós

            Now translate:
            """
            
            for idx, text in enumerate(cleaned_chunk, start=1):
                prompt += f"{idx}. {text}\n"

            try:
                response = model.generate_content(prompt)
                raw = response.text.strip()

                print("[INFO] Gemini raw:\n", raw)

                translated_lines = [line[8:].strip() for line in raw.splitlines() if ">>>" in line]

                if len(translated_lines) == len(cleaned_chunk):
                    return translated_lines
                else:
                    print(f"[WARN] Mismatch: expected {len(cleaned_chunk)} but got {len(translated_lines)}")

            except Exception as e:
                print(f"[ERR] Gemini error: {e}")

            time.sleep(1.5)

        # return untranslated if we keep splitting and it fails to translate even a one-liner
        if len(chunk) == 1:
            print("[WARN] Final fallback failed on single line; returning original")
            return chunk

        # split in half after max # of retries attempted with current chunk size
        print(f"[INFO] Splitting chunk of {len(chunk)} into two...")
        mid = len(chunk) // 2
        first_half = translate_chunk(chunk[:mid], depth + 1)
        second_half = translate_chunk(chunk[mid:], depth + 1)

        return first_half + second_half

    for i in range(0, len(text_list), batch_size):
        chunk = text_list[i:i + batch_size]
        translated = translate_chunk(chunk)
        if not translated:
            print(f"[ERR] Could not translate batch {i}-{i+len(chunk)}.")
            return None
        all_translations.extend(translated)

    return all_translations


"""
the glue
"""

@app.route('/translate', methods=['POST'])
def translate_webpage():
    data = request.get_json()
    url = data.get('url')
    lang = data.get('lang')

    # validate url
    if not url or not lang:
        return jsonify({"error": "Missing URL or language prompt"}), 400

    try:
        # download the page + parse html
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        base_tag = soup.new_tag("base", href=url)
        if soup.head:
            soup.head.insert(0, base_tag)

        # iterate through all texts and translate!
        text_nodes = []

        for tag in soup.body.find_all(text=True):
            skip_tags = ['script', 'style', 'meta', 'noscript', 'head', 'title', 'iframe']
            skip_if_contains = ['yoast', 'gtag', 'analytics', 'pixel', 'robots', 'snippet']
            text = tag.strip().lower()
            if (
            text and
            tag.parent.name not in skip_tags and
            not any(skip_word in text for skip_word in skip_if_contains)
            ): # avoid empty/invis/misc bs
                text_nodes.append(tag)

        original_texts = [t.strip() for t in text_nodes]

        translations = mass_translate(original_texts, lang)

        if not translations:
            return jsonify({ "error": "Translation failed or was incomplete." }), 500

        if translations and len(translations) != len(text_nodes):
            print("[ERR] Line mismatch: Gemini returned ", len(translations), " translations but ", len(text_nodes), " were requested.")
            return jsonify({"error": "Mismatch between translated and original text count."}), 500
        
        if translations and len(translations) == len(text_nodes):
            for original, translated in zip(text_nodes, translations):
                original.replace_with(translated)

        if "background-color" not in str(soup):
            soup.body['style'] = 'background-color: white;'

        # translated page
        return jsonify({ "html": str(soup) })

    except Exception as e:
        # oops!
        return jsonify({"error": str(e)}), 500
    
@app.route('/text-translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text')
    lang = data.get('lang')
    return jsonify({ "translation": str(translate(text, lang)) })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text')
def text():
    return render_template('text.html')

if __name__ == '__main__':
    app.run(debug=True)