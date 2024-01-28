from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_answer(context, question):
    input_text = context + " " + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(input_ids, max_length=100, no_repeat_ngram_size=2, top_k=50, temperature=0.7)
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_answer

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        context = request.form["context"]
        question = request.form["question"]
        answer = generate_answer(context, question)

        return render_template("index.html", answer=answer, context=context, question=question)

    return render_template("index.html", answer=None, context=None, question=None)

if __name__ == "__main__":
    app.run(debug=True)
