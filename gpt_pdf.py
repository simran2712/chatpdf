from transformers import GPT2LMHeadModel, GPT2Tokenizer
import fitz

def generate_questions(context):
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Tokenize the input context
    input_ids = tokenizer.encode(context, return_tensors="pt")

    # Generate questions using the model
    output = model.generate(input_ids, max_length=100, num_questions=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode and print the generated questions
    generated_questions = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Questions:")
    print(generated_questions)

# Example context
# context = """
# The quick brown fox jumps over the lazy dog. In a small town, there was a bakery that specialized in delicious pastries.
# """

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    return text
pdf_path = "48lawsofpower.pdf"

# Extract text from the PDF
context = extract_text_from_pdf(pdf_path)
print(context)

# Generate questions based on the given context
generate_questions(context)
