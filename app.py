from flask import Flask, render_template, request, jsonify, send_file
import os
import re
import string
import html
import traceback
import logging
from werkzeug.utils import secure_filename
import pdfplumber  # Using pdfplumber instead of PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer  # Add sentence-transformers

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable noisy third-party loggers
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Store the extracted text and keywords
pdf_text = ""
keywords = []
paragraphs = []
sections = {}

# Initialize sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentence transformer model: {str(e)}")
    logger.error(traceback.format_exc())

def extract_text_from_pdf(pdf_path):
    """Extract text from uploaded PDF file"""
    logger.info(f"Extracting text from PDF: {pdf_path}")
    try:
        # Check if file exists and is readable
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
            
        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF file size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("PDF file is empty")
            raise ValueError("PDF file is empty")
            
        # Try to open the file to ensure it's not corrupted
        with open(pdf_path, 'rb') as test_file:
            test_data = test_file.read(1024)  # Read first 1KB
            logger.info(f"Successfully read first 1KB of PDF file")
        
        # Extract text using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"PDF has {len(pdf.pages)} pages")
            text = ""
            
            for i, page in enumerate(pdf.pages):
                try:
                    logger.info(f"Processing page {i+1}/{len(pdf.pages)}")
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        logger.info(f"Extracted {len(page_text)} characters from page {i+1}")
                    else:
                        logger.warning(f"No text extracted from page {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {i+1}: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Continue with next page instead of failing completely
                    continue
        
        if not text.strip():
            logger.error("No text was extracted from any page")
            # Return a placeholder message instead of raising an exception
            return "No text could be extracted from this PDF. It may be scanned or contain only images."
            
        logger.info(f"Extracted {len(text)} characters of text in total")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def preprocess_text(text):
    """Split text into paragraphs and clean it"""
    logger.info("Preprocessing text")
    try:
        # Split by double newlines to get paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        logger.info(f"Split text into {len(paragraphs)} raw paragraphs")
        
        # Clean paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            # Replace single newlines with spaces
            cleaned = re.sub(r'\n', ' ', para)
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if cleaned and len(cleaned) > 50:  # Only add non-empty paragraphs with meaningful content
                cleaned_paragraphs.append(cleaned)
        
        logger.info(f"Retained {len(cleaned_paragraphs)} cleaned paragraphs")
        return cleaned_paragraphs
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def identify_sections(paragraphs):
    """Identify sections in the document based on headings"""
    logger.info("Identifying sections in the document")
    try:
        sections = {}
        current_section = "Introduction"
        section_content = []
        
        # Common section heading patterns
        section_patterns = [
            r'^[A-Z][A-Za-z\s]+:',  # "Section Name:"
            r'^[0-9]+\.\s+[A-Z][A-Za-z\s]+',  # "1. Section Name"
            r'^[A-Z][A-Z\s]+$',  # "SECTION NAME"
        ]
        
        for para in paragraphs:
            # Check if paragraph is a section heading
            is_heading = False
            for pattern in section_patterns:
                if re.match(pattern, para):
                    # Save previous section
                    if section_content:
                        sections[current_section] = ' '.join(section_content)
                        logger.info(f"Identified section: {current_section} with {len(section_content)} paragraphs")
                    
                    # Start new section
                    current_section = para
                    section_content = []
                    is_heading = True
                    break
            
            if not is_heading:
                section_content.append(para)
        
        # Add the last section
        if section_content:
            sections[current_section] = ' '.join(section_content)
            logger.info(f"Identified section: {current_section} with {len(section_content)} paragraphs")
        
        logger.info(f"Identified {len(sections)} sections in total")
        return sections
    except Exception as e:
        logger.error(f"Error identifying sections: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def extract_keywords(text, top_n=10):
    """Extract top keywords using TF-IDF"""
    logger.info("Extracting keywords using TF-IDF")
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=100,
            ngram_range=(1, 2)  # Include bigrams
        )
        
        # Fit and transform the text
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Sort by score
        sorted_indices = tfidf_scores.argsort()[::-1]
        
        # Get top keywords
        top_keywords = [feature_names[idx] for idx in sorted_indices[:top_n]]
        
        logger.info(f"Extracted {len(top_keywords)} keywords: {', '.join(top_keywords)}")
        return top_keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def format_response(text):
    """Format the response text to be more readable"""
    # Split into sentences using regex instead of NLTK
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Group sentences into paragraphs (3-4 sentences per paragraph)
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        if len(current_paragraph) >= 3:  # Create a new paragraph after 3 sentences
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    # Add any remaining sentences as a paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Join paragraphs with double newlines for spacing
    formatted_text = '\n\n'.join(paragraphs)
    
    return formatted_text

def preprocess_question(question):
    """Preprocess the question for better matching"""
    # Simple preprocessing - lowercase and remove extra whitespace
    return question.lower().strip()

def get_question_intent(question):
    """Determine the main intent of the question"""
    question_lower = question.lower()
    
    # Define intent categories with their keywords
    intent_categories = {
        'communication': ['communicate', 'communication', 'chat', 'message', 'talk', 'discuss', 'conversation', 'contact'],
        'team': ['team', 'group', 'collaborate', 'collaboration', 'member', 'staff', 'personnel'],
        'schedule': ['schedule', 'time', 'date', 'deadline', 'when', 'timeline'],
        'budget': ['budget', 'cost', 'price', 'expense', 'financial', 'money', 'payment'],
        'features': ['feature', 'functionality', 'capability', 'function', 'able to', 'can do'],
        'requirements': ['requirement', 'need', 'necessary', 'must have', 'should have'],
        'rewards': ['reward', 'incentive', 'bonus', 'compensation', 'recognition']
    }
    
    # Count matches for each category
    intent_scores = {}
    for category, keywords in intent_categories.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        if score > 0:
            intent_scores[category] = score
    
    # Log the detected intents
    logger.info(f"Question intent scores: {intent_scores}")
    
    # Return the category with highest score, or None if no matches
    if intent_scores:
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Primary question intent: {primary_intent}")
        return primary_intent
    return None

def find_relevant_paragraphs(question, paragraphs, top_k=5):
    """Find the most relevant paragraphs using sentence transformers"""
    global sections, model
    
    logger.info(f"Finding relevant paragraphs for question: {question}")
    
    # Preprocess the question
    processed_question = preprocess_question(question)
    
    # Extract key terms from the question
    question_terms = processed_question.split()
    
    # Get question intent
    question_intent = get_question_intent(question)
    logger.info(f"Question intent: {question_intent}")
    
    # Log the question terms for debugging
    logger.info(f"Question terms: {question_terms}")
    
    # Additional keywords based on question intent
    intent_keywords = {
        'communication': ['communication', 'messaging', 'chat', 'collaboration', 'tools', 'platform', 'contact', 'discuss'],
        'team': ['team', 'collaboration', 'group', 'members', 'colleagues', 'department'],
        'rewards': ['rewards', 'incentives', 'bonuses', 'recognition', 'achievements', 'performance']
    }
    
    if question_intent and question_intent in intent_keywords:
        question_terms.extend(intent_keywords[question_intent])
        logger.info(f"Added intent keywords: {intent_keywords[question_intent]}")
    
    # Check if question is about a specific section or topic
    section_match = None
    for section_name in sections.keys():
        if any(term in section_name.lower() for term in question_terms):
            section_match = section_name
            logger.info(f"Found matching section: {section_name}")
            break
    
    # If question matches a section, prioritize that section
    if section_match:
        # Split section content into paragraphs
        section_paragraphs = preprocess_text(sections[section_match])
        if section_paragraphs:
            # If we have enough paragraphs in the section, use only those
            if len(section_paragraphs) >= 3:
                paragraphs = section_paragraphs
                logger.info(f"Using {len(paragraphs)} paragraphs from section: {section_match}")
    
    # Filter paragraphs that contain at least one question term
    relevant_paragraphs = []
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        if any(term in paragraph_lower for term in question_terms):
            relevant_paragraphs.append(paragraph)
    
    # If we found relevant paragraphs, use those; otherwise use all paragraphs
    if relevant_paragraphs:
        logger.info(f"Found {len(relevant_paragraphs)} paragraphs containing question terms")
        logger.info(f"First relevant paragraph: {relevant_paragraphs[0][:100]}...")
        paragraphs = relevant_paragraphs
    
    try:
        # Encode the question and paragraphs using sentence transformers
        question_embedding = model.encode([processed_question])[0]
        paragraph_embeddings = model.encode(paragraphs)
        
        # Calculate cosine similarity between question and paragraphs
        similarities = cosine_similarity([question_embedding], paragraph_embeddings)[0]
        
        # Get top-k most similar paragraphs
        top_indices = similarities.argsort()[::-1][:top_k]
        
        # Log similarity scores for debugging
        for i, idx in enumerate(top_indices):
            logger.info(f"Top match {i+1}: Score {similarities[idx]:.4f}, Text: {paragraphs[idx][:100]}...")
        
        # Return the most relevant paragraphs
        return [paragraphs[idx] for idx in top_indices]
    except Exception as e:
        logger.error(f"Error finding relevant paragraphs: {str(e)}")
        logger.error(traceback.format_exc())
        # Fallback if encoding fails
        logger.info("Falling back to TF-IDF vectorization")
        try:
            # Create a vectorizer for paragraphs and question
            vectorizer = TfidfVectorizer(stop_words='english')
            
            # Create document-term matrix
            tfidf_matrix = vectorizer.fit_transform(paragraphs + [processed_question])
            
            # Calculate cosine similarity between question and paragraphs
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            
            # Get top-k most similar paragraphs
            top_indices = similarities.argsort()[::-1][:top_k]
            
            # Return the most relevant paragraphs
            return [paragraphs[idx] for idx in top_indices]
        except:
            # Ultimate fallback
            return paragraphs[:min(top_k, len(paragraphs))]

def answer_question(question, relevant_paragraphs):
    """Generate an answer based on the relevant paragraphs"""
    global model
    
    # Join paragraphs to create context
    context = " ".join(relevant_paragraphs)
    
    # Get question intent
    question_intent = get_question_intent(question)
    
    # Check if the answer is actually relevant to the question
    # Extract key terms from the question
    question_terms = preprocess_question(question).split()
    
    # Add additional search terms based on intent
    if question_intent == 'communication':
        question_terms.extend(['communicate', 'communication', 'messaging', 'contact', 'platform', 'chat'])
    elif question_intent == 'team':
        question_terms.extend(['team', 'collaboration', 'members', 'together'])
    
    # Check for specific question patterns and provide custom responses
    if question_intent == 'communication' and 'team' in preprocess_question(question):
        # Look specifically for communication-related information
        communication_sentences = []
        communication_terms = ['communicate', 'communication', 'messaging', 'chat', 'discussion', 'conversation', 
                               'platform', 'contact', 'email', 'meeting', 'call', 'conference', 'notify']
        
        # Extract sentences from context
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Filter for communication-related sentences
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in communication_terms):
                communication_sentences.append(sentence)
        
        # If found communication-related sentences, use them for the answer
        if communication_sentences:
            return " ".join(communication_sentences)
        else:
            # If no specific communication info found, provide a generic response
            return """I couldn't find specific information about team communication methods in the proposal. 
            
The proposal doesn't appear to detail internal team communication protocols. You might want to check if there's a separate communication plan or team collaboration document. Based on similar projects, teams typically use a combination of:

1. Project management software (like Jira, Asana, or Trello)
2. Team messaging platforms (like Slack or Microsoft Teams)
3. Regular video meetings (via Zoom or Teams)
4. Email for formal communications
5. Documentation repositories (like Confluence or SharePoint)

Consider asking the project manager for the specific communication channels planned for this project."""
    
    # For direct questions, try to extract specific sentences
    question_words = ["what", "who", "when", "where", "why", "how"]
    
    # Check if it's a direct question
    is_direct_question = any(question.lower().startswith(qw) for qw in question_words)
    
    if is_direct_question:
        try:
            # Split context into sentences
            sentences = re.split(r'(?<=[.!?])\s+', context)
            
            # Filter sentences that contain at least one question term
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(term in sentence_lower for term in question_terms):
                    relevant_sentences.append(sentence)
            
            # If we found relevant sentences, use those; otherwise use all sentences
            if relevant_sentences:
                logger.info(f"Found {len(relevant_sentences)} sentences containing question terms")
                sentences = relevant_sentences
            
            try:
                # Encode the question and sentences using sentence transformers
                question_embedding = model.encode([question])[0]
                sentence_embeddings = model.encode(sentences)
                
                # Calculate cosine similarity
                similarities = cosine_similarity([question_embedding], sentence_embeddings)[0]
                
                # Get the most relevant sentence
                most_relevant_idx = similarities.argmax()
                most_relevant_sentence = sentences[most_relevant_idx]
                
                # Get the next sentence for context if available
                next_idx = most_relevant_idx + 1
                if next_idx < len(sentences):
                    answer = most_relevant_sentence + " " + sentences[next_idx]
                else:
                    answer = most_relevant_sentence
                
                # If the answer doesn't seem relevant enough, add a disclaimer
                if max(similarities) < 0.2 or len(relevant_sentences) == 0:
                    logger.warning(f"Low relevance score: {max(similarities):.4f}")
                    
                    # For communication questions specifically
                    if question_intent == 'communication':
                        return """I couldn't find specific information about team communication methods in the proposal. 
                        
The proposal doesn't appear to detail internal team communication protocols. You might want to check if there's a separate communication plan or team collaboration document. Based on similar projects, teams typically use a combination of:

1. Project management software (like Jira, Asana, or Trello)
2. Team messaging platforms (like Slack or Microsoft Teams)
3. Regular video meetings (via Zoom or Teams)
4. Email for formal communications
5. Documentation repositories (like Confluence or SharePoint)

Consider asking the project manager for the specific communication channels planned for this project."""
                    else:
                        answer = "I couldn't find specific information about that in the proposal. Here's the closest match I found:\n\n" + answer
            except Exception as e:
                logger.error(f"Error with sentence transformer in answer_question: {str(e)}")
                # Fallback to CountVectorizer if sentence transformer fails
                # Vectorize the question and sentences
                vectorizer = CountVectorizer(stop_words='english')
                
                # Create document-term matrix
                vectors = vectorizer.fit_transform([question] + sentences)
                
                # Calculate cosine similarity
                similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
                
                # Get the most relevant sentence
                most_relevant_idx = similarities.argmax()
                most_relevant_sentence = sentences[most_relevant_idx]
                
                # Get the next sentence for context if available
                next_idx = most_relevant_idx + 1
                if next_idx < len(sentences):
                    answer = most_relevant_sentence + " " + sentences[next_idx]
                else:
                    answer = most_relevant_sentence
                    
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to returning the first paragraph
            return relevant_paragraphs[0]
    else:
        # For non-direct questions, return the most relevant paragraph
        return relevant_paragraphs[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global pdf_text, keywords, paragraphs, sections
    
    logger.info("Upload endpoint called")
    
    if 'pdf' not in request.files:
        logger.warning("No file provided in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        logger.warning("Empty filename provided")
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to {filepath}")
            file.save(filepath)
            
            # Check if file was saved correctly
            if not os.path.exists(filepath):
                logger.error(f"File was not saved correctly at {filepath}")
                return jsonify({'error': 'File could not be saved'}), 500
                
            logger.info(f"File saved successfully. Size: {os.path.getsize(filepath)} bytes")
            
            # Extract text from PDF
            logger.info("Starting PDF text extraction")
            try:
                pdf_text = extract_text_from_pdf(filepath)
                logger.info(f"Extracted {len(pdf_text)} characters of text")
            except Exception as e:
                logger.error(f"Error in extract_text_from_pdf: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f"Error extracting text from PDF: {str(e)}"}), 500
            
            # Process text into paragraphs for better context
            logger.info("Processing text into paragraphs")
            try:
                paragraphs = preprocess_text(pdf_text)
                logger.info(f"Processed into {len(paragraphs)} paragraphs")
            except Exception as e:
                logger.error(f"Error in preprocess_text: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f"Error preprocessing text: {str(e)}"}), 500
            
            # Identify sections in the document
            logger.info("Identifying sections in the document")
            try:
                sections = identify_sections(paragraphs)
                logger.info(f"Identified {len(sections)} sections")
            except Exception as e:
                logger.error(f"Error in identify_sections: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f"Error identifying sections: {str(e)}"}), 500
            
            # Extract keywords using TF-IDF
            logger.info("Extracting keywords")
            try:
                keywords = extract_keywords(pdf_text)
                logger.info(f"Extracted {len(keywords)} keywords: {keywords}")
            except Exception as e:
                logger.error(f"Error in extract_keywords: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f"Error extracting keywords: {str(e)}"}), 500
            
            logger.info("File processed successfully")
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'keywords': keywords
            })
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing file: {error_msg}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f"Error processing file: {error_msg}"}), 500
    else:
        logger.warning("Invalid file type")
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global paragraphs
    
    if not paragraphs:
        return jsonify({'error': 'No PDF has been processed yet'}), 400
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Find relevant paragraphs
        relevant_paragraphs = find_relevant_paragraphs(question, paragraphs)
        
        # Generate answer
        answer_text = answer_question(question, relevant_paragraphs)
        
        # If answer is too short, provide more context
        if len(answer_text) < 100:
            # Include additional context from relevant paragraphs
            additional_context = relevant_paragraphs[0] if relevant_paragraphs else ""
            if answer_text not in additional_context:
                answer_text = f"{answer_text}\n\nAdditional context: {additional_context}"
        
        # Format the response for better readability
        formatted_text = format_response(answer_text)
        
        # Format the final response
        final_answer = f"Based on the proposal, here's what I found:\n\n{formatted_text}"
        
        return jsonify({'answer': final_answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use environment variables for configuration in production
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 