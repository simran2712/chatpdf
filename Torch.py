import fitz
import torch
import torch.nn as nn
import torch.optim as optim

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    return text
pdf_path = "48lawsofpower.pdf"

pdf_text = extract_text_from_pdf(pdf_path)
print(pdf_text)

def preprocess_text(text):
    return ''.join([c.lower() for c in text if c.isalpha() or c.isspace()])

preprocessed_text = preprocess_text(pdf_text)

chars = sorted(list(set(preprocessed_text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

input_size = len(chars)
hidden_size = 128
output_size = len(chars)

model = CharRNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

seq_length = 100
num_epochs = 1000

for epoch in range(num_epochs):
    for i in range(0, len(preprocessed_text) - seq_length, seq_length):
        inputs = [char_to_index[ch] for ch in preprocessed_text[i:i+seq_length]]
        targets = [char_to_index[ch] for ch in preprocessed_text[i+1:i+seq_length+1]]

        inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0)
        targets = torch.tensor(targets, dtype=torch.long).unsqueeze(0)

        optimizer.zero_grad()
        hidden = None

        output, hidden = model(inputs, hidden)
        loss = criterion(output.view(-1, output_size), targets.view(-1))
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def generate_text(model, start_char='a', length=200):
    model.eval()
    result = start_char
    char_index = char_to_index[start_char]
    input_seq = torch.tensor([char_index], dtype=torch.long).unsqueeze(0)
    hidden = None

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        probabilities = nn.functional.softmax(output.view(-1), dim=0)
        predicted_index = torch.multinomial(probabilities, 1).item()
        result += index_to_char[predicted_index]
        input_seq[0][0] = predicted_index

    return result

generated_text = generate_text(model, start_char='a', length=500)
print(generated_text)
