from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_model():
    """Load your fine-tuned model"""
    print("🔄 Loading your fine-tuned model...")
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')
    model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
    print("✅ Model loaded successfully!")
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=100):
    """Generate text from a prompt"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    # Load the model
    model, tokenizer = load_model()
    
    print("\n🎯 GPT-2 Text Generator Ready!")
    print("Type your prompts below (or 'quit' to exit):")
    
    while True:
        prompt = input("\n📝 Enter prompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not prompt:
            continue
            
        print("🤖 Generating...")
        generated = generate_text(prompt, model, tokenizer)
        print(f"📄 Result: {generated}")

if __name__ == "__main__":
    main()