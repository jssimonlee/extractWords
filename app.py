import os
import re
import json
import zipfile
import urllib.request
import urllib.parse
import pypdf
from flask import Flask, request, jsonify, render_template, send_file, make_response
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from cefrpy import CEFRAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------------------------------------------------------
# NLTK Automatic Initialization
# -----------------------------------------------------------------------------
def init_nltk():
    print("Initializing NLTK resources...")
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Downloading NLTK resource: {name}")
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Error downloading {name}: {e}")
    print("NLTK initialization finished.")

init_nltk()

# -----------------------------------------------------------------------------
# Translation Cache Setup
# -----------------------------------------------------------------------------
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'translation_cache.json')
translation_cache = {}

def load_translation_cache():
    global translation_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                translation_cache = json.load(f)
        except Exception as e:
            print(f"Error loading translation cache: {e}")
            translation_cache = {}

def save_translation_cache():
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(translation_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving translation cache: {e}")

load_translation_cache()

# -----------------------------------------------------------------------------
# NLP Engine Helpers
# -----------------------------------------------------------------------------
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def get_cefrpy_pos(treebank_tag):
    if treebank_tag.startswith('V') or treebank_tag == 'MD':
        return 'VB'
    elif treebank_tag.startswith('J'):
        return 'JJ'
    elif treebank_tag.startswith('R'):
        return 'RB'
    elif treebank_tag.startswith('N'):
        return 'NN'
    return 'NN'

def get_detailed_pos(treebank_tag, word):
    tag = treebank_tag
    word_lower = word.lower()
    
    # 1. 조동사 (Modal Verb)
    if tag == 'MD':
        return 'MD'
    # 2. 동사류
    elif tag.startswith('V'):
        return 'VB'
    # 3. 형용사류
    elif tag.startswith('J'):
        return 'JJ'
    # 4. 부사류
    elif tag.startswith('R') or tag == 'WRB':
        return 'RB'
    # 5. 명사류
    elif tag.startswith('N'):
        return 'NN'
    # 6. 대명사류
    elif tag in ['PRP', 'PRP$', 'WP', 'WP$', 'EX']:
        return 'PRON'
    # 7. 등위접속사
    elif tag == 'CC':
        return 'CONJ'
    # 8. 전치사 및 종속접속사 (IN)
    elif tag == 'IN':
        conjunctions = {
            'although', 'though', 'while', 'because', 'if', 'unless', 'since', 
            'until', 'before', 'after', 'whereas', 'once', 'whether', 'as', 'than'
        }
        if word_lower in conjunctions:
            return 'CONJ'
        else:
            return 'PREP'
    # 9. 전치사 TO
    elif tag == 'TO' and word_lower not in ['to']:
        return 'PREP'
    # 10. 기타 (한정사 DT, 감탄사 UH, 파티클 RP 등)
    return 'ETC'

def determine_cefr_level(analyzer, lemma, pos_tag):
    cefr_pos = get_cefrpy_pos(pos_tag)
    # Try POS-specific lookup
    level = analyzer.get_word_pos_level_CEFR(lemma, cefr_pos)
    if not level:
        # Fall back to average lookup
        level = analyzer.get_average_word_level_CEFR(lemma)
    
    if level:
        return level.name
    return None


def is_proper_noun(word, is_first_word, pos_tag, analyzer):
    # Acronyms (e.g. USA, FBI) are proper nouns
    if word.isupper() and len(word) > 1:
        return True
        
    # Check if NLTK tagged it as a proper noun
    if pos_tag in ['NNP', 'NNPS']:
        # If it's the first word of a sentence and exists in lowercase in CEFR database, it's a common word (e.g., "Although")
        if is_first_word and analyzer.is_word_in_database(word.lower()):
            return False
        return True
        
    # If capitalized, not at the start of sentence, and not a common word in the database
    if len(word) > 0 and word[0].isupper() and not is_first_word:
        if not analyzer.is_word_in_database(word.lower()):
            return True
            
    return False

# -----------------------------------------------------------------------------
# Core Parser & Analyzer
# -----------------------------------------------------------------------------
def extract_epub_text_and_sentences(epub_path):
    sentences = []
    text_content = []
    
    try:
        with zipfile.ZipFile(epub_path, 'r') as epub_zip:
            # 1. Parse container.xml to locate OPF file
            container_path = 'META-INF/container.xml'
            opf_path = None
            
            if container_path in epub_zip.namelist():
                container_data = epub_zip.read(container_path)
                soup = BeautifulSoup(container_data, 'xml')
                rootfile = soup.find('rootfile')
                if rootfile and rootfile.get('full-path'):
                    opf_path = rootfile.get('full-path')
            
            # If opf_path found, parse it to extract HTML in spine reading order
            html_files = []
            if opf_path and opf_path in epub_zip.namelist():
                opf_data = epub_zip.read(opf_path)
                opf_soup = BeautifulSoup(opf_data, 'xml')
                
                # Extract manifest item map
                manifest = {}
                for item in opf_soup.find_all('item'):
                    item_id = item.get('id')
                    item_href = item.get('href')
                    item_media = item.get('media-type')
                    if item_id and item_href and item_media and ('html' in item_media or 'xhtml' in item_media):
                        manifest[item_id] = item_href
                
                # Get spine items in order
                spine_items = []
                for itemref in opf_soup.find_all('itemref'):
                    idref = itemref.get('idref')
                    if idref in manifest:
                        spine_items.append(manifest[idref])
                
                # Resolve paths relative to OPF base directory
                opf_dir = os.path.dirname(opf_path)
                for href in spine_items:
                    if opf_dir:
                        full_href = os.path.normpath(os.path.join(opf_dir, href)).replace('\\', '/')
                    else:
                        full_href = href
                        
                    if full_href in epub_zip.namelist():
                        html_files.append(full_href)
                    else:
                        # Fallback for url-encoded paths
                        decoded_href = urllib.parse.unquote(full_href)
                        if decoded_href in epub_zip.namelist():
                            html_files.append(decoded_href)
            
            # Fallback: scan for all html files if spine parsing failed
            if not html_files:
                html_files = [
                    f for f in epub_zip.namelist()
                    if f.endswith(('.html', '.xhtml', '.htm'))
                ]
                html_files.sort()
                
            # 2. Extract visible text from each document file
            for file_path in html_files:
                try:
                    html_data = epub_zip.read(file_path)
                    soup = BeautifulSoup(html_data, 'html.parser')
                    
                    # Remove non-readable nodes
                    for node in soup(['script', 'style', 'head', 'title', 'meta']):
                        node.decompose()
                        
                    paras = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                    file_text = ""
                    if paras:
                        file_text = "\n".join([p.get_text().strip() for p in paras if p.get_text().strip()])
                    else:
                        file_text = soup.get_text()
                        
                    if file_text:
                        text_content.append(file_text)
                except Exception as e:
                    print(f"Error parsing chapter {file_path}: {e}")
                    
        full_text = "\n\n".join(text_content)
        # Tokenize sentences
        sentences = nltk.sent_tokenize(full_text)
        # Standardize whitespace
        sentences = [s.strip().replace('\r', '').replace('\n', ' ') for s in sentences if s.strip()]
        
    except Exception as e:
        print(f"Error reading EPUB {epub_path}: {e}")
        
    return sentences

def extract_pdf_text_and_sentences(pdf_path):
    sentences = []
    text_content = []
    
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            except Exception as e:
                print(f"Error parsing PDF page {page_num}: {e}")
                
        full_text = "\n\n".join(text_content)
        # Tokenize sentences
        sentences = nltk.sent_tokenize(full_text)
        # Standardize whitespace
        sentences = [s.strip().replace('\r', '').replace('\n', ' ') for s in sentences if s.strip()]
        
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        
    return sentences

def analyze_vocabulary(sentences):
    analyzer = CEFRAnalyzer()
    lemmatizer = WordNetLemmatizer()
    
    CONTRACTION_GARBAGE = {
        # 1. Apostrophe split leftovers & prefix contraction remnants
        'nt', 're', 'll', 've', 'd', 'm', 's',
        'don', 'doesn', 'didn', 'wasn', 'weren', 'hasn', 'haven', 'hadn', 
        'isn', 'shouldn', 'wouldn', 'couldn', 'mustn', 'aren', 'needn', 
        'daren', 'oughtn', 'wo', 'sha',
        # 2. Apostrophe omitted/fused invalid words
        'dont', 'theyre', 'youre', 'ive', 'hes', 'shes', 'cant', 'wont', 
        'wouldnt', 'couldnt', 'shouldnt', 'havent', 'hasnt', 'didnt', 
        'arent', 'werent', 'youll', 'theyll', 'shell', 'hell', 'its', 
        'im', 'weve', 'theyve', 'youve', 'hadnt', 'isnt', 'wasnt', 'werent',
        'doesnt', 'dont', 'shant', 'mustnt'
    }
    
    word_data = {}
    word_pattern = re.compile(r'^[a-zA-Z]+$')
    
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if not words:
            continue
            
        try:
            tagged_words = nltk.pos_tag(words)
        except Exception as e:
            tagged_words = [(w, 'NN') for w in words]
            
        for i, (word, pos_tag) in enumerate(tagged_words):
            # Keep letters only, filter out punctuation/symbols and single letter words
            if not word_pattern.match(word) or len(word) < 2:
                continue
                
            is_first = (i == 0)
            
            # Smart filter for proper nouns/names/places
            if is_proper_noun(word, is_first, pos_tag, analyzer):
                continue
                
            # Get lemmatized form
            wn_pos = get_wordnet_pos(pos_tag)
            lemma = lemmatizer.lemmatize(word.lower(), pos=wn_pos)
            
            if lemma in CONTRACTION_GARBAGE:
                continue
                
            if len(lemma) < 2 or not lemma.isalpha():
                continue
                
            # Classify CEFR level
            level = determine_cefr_level(analyzer, lemma, pos_tag)
            if not level:
                level = 'U' # Unclassified (Highly Advanced / Special Jargon)
                
            if lemma not in word_data:
                word_data[lemma] = {
                    'word': lemma,
                    'level': level,
                    'count': 1,
                    'context': sentence,
                    'pos': get_detailed_pos(pos_tag, word)
                }
            else:
                word_data[lemma]['count'] += 1
                
    # Sort descending by frequency
    sorted_words = sorted(word_data.values(), key=lambda x: x['count'], reverse=True)
    return sorted_words

# -----------------------------------------------------------------------------
# Flask API Endpoints
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_epub():
    # 1. Check if JSON payload is sent (Direct Text Paste / Clipboard)
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
        filename = data.get('filename', '입력한 텍스트')
        if not text.strip():
            return jsonify({'error': 'No text content provided'}), 400
            
        try:
            # Tokenize sentences directly from pasted text
            sentences = nltk.sent_tokenize(text)
            sentences = [s.strip().replace('\r', '').replace('\n', ' ') for s in sentences if s.strip()]
            
            if not sentences:
                return jsonify({'error': 'Failed to parse any sentences from the text.'}), 400
                
            words = analyze_vocabulary(sentences)
            return jsonify({
                'filename': filename,
                'total_sentences': len(sentences),
                'unique_words_count': len(words),
                'words': words
            })
        except Exception as e:
            print(f"Error processing pasted text: {e}")
            return jsonify({'error': f"Internal server error: {str(e)}"}), 500

    # 2. Otherwise handle Multipart file upload (.epub, .txt)
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    fn_lower = file.filename.lower()
    if not (fn_lower.endswith('.epub') or fn_lower.endswith('.txt') or fn_lower.endswith('.pdf')):
        return jsonify({'error': 'Invalid file format. Only EPUB, TXT, and PDF files are supported.'}), 400
        
    try:
        if fn_lower.endswith('.txt'):
            # Text file: read and decode directly
            text_data = file.read().decode('utf-8', errors='ignore')
            sentences = nltk.sent_tokenize(text_data)
            sentences = [s.strip().replace('\r', '').replace('\n', ' ') for s in sentences if s.strip()]
        elif fn_lower.endswith('.pdf'):
            # PDF file: save to temp and parse
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(temp_path)
            
            sentences = extract_pdf_text_and_sentences(temp_path)
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            # EPUB file: existing flow
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(temp_path)
            
            sentences = extract_epub_text_and_sentences(temp_path)
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not sentences:
            return jsonify({'error': 'Failed to extract text. The file may be empty or DRM-protected.'}), 400
            
        words = analyze_vocabulary(sentences)
            
        return jsonify({
            'filename': file.filename,
            'total_sentences': len(sentences),
            'unique_words_count': len(words),
            'words': words
        })
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

@app.route('/api/translate', methods=['GET'])
def translate():
    word = request.args.get('word', '').lower().strip()
    if not word:
        return jsonify({'error': 'No word provided'}), 400
        
    translated = translate_word_to_korean(word)
    return jsonify({
        'word': word,
        'translation': translated
    })

def translate_word_to_korean(word):
    word = word.lower().strip()
    if word in translation_cache:
        return translation_cache[word]
        
    # Free translation via MyMemory API
    try:
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(word)}&langpair=en|ko"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))
            translated_text = data.get('responseData', {}).get('translatedText', '')
            
            if translated_text:
                # Remove sentence-level translation bugs (e.g. if it translates word as full sentence,
                # we just take the main terms or preserve it)
                cleaned = translated_text.strip()
                translation_cache[word] = cleaned
                save_translation_cache()
                return cleaned
    except Exception as e:
        print(f"MyMemory translation failed for '{word}': {e}")
        
    return ""

if __name__ == '__main__':
    print("Starting LingoExtractor Web Server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
