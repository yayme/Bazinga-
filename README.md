# 🤖 Sarcasm Detector with BERT

> “Just what I needed at this hour! A flat tire. Amazing”  
**→ Bazinga! (probability: 0.829)**

Yes — like **Dr. Sheldon Cooper** from *The Big Bang Theory*, this model detects sarcasm and proudly shouts **"Bazinga!"** when it finds some.

---

## 🧠 What Is This?

A simple sarcasm detector built using **PyTorch** and **BERT**, trained on the [TweetEval: Irony](https://huggingface.co/datasets/tweet_eval) dataset. Given a tweet or sentence, it returns the probability that it’s sarcastic — and celebrates with a Bazinga if it is.

---

## 🚀 Key Features

- 🧾 Uses `bert-base-uncased` for understanding context
- 🔍 Tokenizes text and applies attention masking
- 🧠 Fine-tuned for binary sarcasm detection
- 📊 Outputs a sarcasm probability (0 to 1)

---

## 💡 How It Works

1. **Dataset**: Uses Hugging Face's `tweet_eval` (irony subset)
2. **Tokenizer**: Converts raw text into BERT-compatible format
3. **Model**: BERT + 1 Linear layer → Sigmoid
4. **Training**: Binary Cross-Entropy Loss over 3 epochs
5. **Inference**: If sarcasm score > 0.5 → **Bazinga!**

---

## 🧪 Example

```python
print_result("Just what I needed at this hour! a flat tire. Amazing")
# → Bazinga! (probability: 0.829)
