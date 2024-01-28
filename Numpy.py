import numpy as np
import fitz

# Step 1: Convert PDF to Text (Assuming you have the text)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    return text
pdf_path = "48lawsofpower.pdf"

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)
print(pdf_text)


# Step 2: Preprocess the Text
def preprocess_text(text):
    return ''.join([c.lower() for c in text if c.isalpha() or c.isspace()])

preprocessed_text = preprocess_text(pdf_text)

# Step 3: Tokenization and Vocabulary
chars = sorted(list(set(preprocessed_text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# Hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 0.01
num_epochs = 500

# Initialize weights
Wxh = np.random.randn(hidden_size, len(chars)) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(len(chars), hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((len(chars), 1))

# Training loop
for epoch in range(num_epochs):
    h_prev = np.zeros((hidden_size, 1))
    inputs = [char_to_index[ch] for ch in preprocessed_text[:-1]]
    targets = [char_to_index[ch] for ch in preprocessed_text[1:]]
    loss = 0

    for i in range(0, len(inputs), seq_length):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        seq_len = min(seq_length, len(inputs) - i)

        # Forward pass
        for t in range(seq_len):
            xs[t] = np.zeros((len(chars), 1))
            xs[t][inputs[i + t]] = 1
            hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
            ys[t] = np.dot(Why, hs[t]) + by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[i + t], 0])

        # Backward pass
        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        dbh, dby = np.zeros_like(bh), np.zeros_like(by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(seq_len)):
            dy = np.copy(ps[t])
            dy[targets[i + t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, xs[t].T)
            dWhh += np.dot(dh_raw, hs[t-1].T)
            dh_next = np.dot(Whh.T, dh_raw)

        # Update weights
        for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
            np.clip(dparam, -5, 5, out=dparam)
            param -= learning_rate * dparam

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

# Generate text
def generate_text(seed_text, length=200):
    h = np.zeros((hidden_size, 1))
    x = np.zeros((len(chars), 1))
    generated_text = seed_text

    for _ in range(length):
        x[char_to_index[seed_text[-1]]] = 1
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        next_char = np.random.choice(chars, p=p.flatten())
        generated_text += next_char
        seed_text = seed_text[1:] + next_char

    return generated_text

# Generate text starting from a seed
seed_text = "your seed text here."
generated_text = generate_text(seed_text, length=500)
print(generated_text)
