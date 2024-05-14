from pickletools import optimize
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Example conversation data
conversations = [
    ("Hello!", "Hi there!"),
    ("How are you?", "I'm doing well, thanks."),
    ("What's the weather like today?", "It's sunny outside."),
    # Add more conversations from your data
]

# Tokenize and format data
inputs = [tokenizer.encode(conversation[0], return_tensors="pt") for conversation in conversations]
outputs = [tokenizer.encode(conversation[1], return_tensors="pt") for conversation in conversations]

# Train the model (fine-tuning)
for input, output in zip(inputs, outputs):
    # Adjust input and output lengths if necessary
    input = input[:, :512]
    output = output[:, :512]
    
    # Forward pass
    outputs = model(input_ids=input, labels=output)
    loss = outputs.loss
    
    # Backward pass and optimization step (pseudo code, actual optimization may vary)
    loss.backward()
    optimize.step()
    optimize.zero_grad()

# Save or deploy the trained model
torch.save(model.state_dict(), "trained_chatbot_model.pth")
