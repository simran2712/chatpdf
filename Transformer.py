import fitz
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    return text

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a new padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Path to your PDF file
pdf_path = "48lawsofpower.pdf"

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Save the extracted text to a text file
with open("pdf_text.txt", "w", encoding="utf-8") as text_file:
    text_file.write(pdf_text)

# Load your custom dataset using the datasets library
dataset = load_dataset('text', data_files='pdf_text.txt')

# Tokenize the dataset with padding and truncation
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, return_tensors='pt')

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./custom_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
    train_dataset=tokenized_datasets["train"],  # Use 'train' instead of 'train_dataset'
)

# Fine-tune the model
trainer.train()
