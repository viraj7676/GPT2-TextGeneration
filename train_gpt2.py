import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

def main():
    print("🚀 Starting GPT-2 Fine-tuning...")
    
    # Check if we have a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💾 Using device: {device}")
    
    # 1. Load tokenizer and model
    print("📝 Loading GPT-2 model and tokenizer...")
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # 2. Load your training data
    print("📚 Loading training data...")
    try:
        with open('train.txt', 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ Loaded {len(texts)} training examples")
    except FileNotFoundError:
        print("❌ Error: train.txt file not found!")
        print("Please create a train.txt file with your training data.")
        return
    
    # 3. Tokenize the data
    print("🔤 Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=128,  # Shorter for faster training
            return_tensors='pt'
        )
    
    # Create dataset
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 4. Set up training arguments
    print("⚙️ Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir='./gpt2-finetuned',
        overwrite_output_dir=True,
        num_train_epochs=2,  # Start with just 2 epochs
        per_device_train_batch_size=2,  # Small batch size for stability
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to=None,  # Disable wandb logging
    )
    
    # 5. Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 6. Initialize trainer
    print("🎯 Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # 7. Start training
    print("🔥 Starting training... This might take a few minutes.")
    print("You'll see training progress below:")
    trainer.train()
    
    # 8. Save the fine-tuned model
    print("💾 Saving the fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained('./gpt2-finetuned')
    print("✅ Model saved to './gpt2-finetuned'")
    
    # 9. Test the model
    print("\n🧪 Testing text generation...")
    test_generation(model, tokenizer)
    
    print("\n🎉 Training completed successfully!")
    print("Your fine-tuned model is ready to use!")

def test_generation(model, tokenizer):
    """Test the fine-tuned model with sample prompts"""
    model.eval()
    
    test_prompts = [
        "The future of",
        "Technology will",
        "Artificial intelligence"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 30,  # Generate 30 more tokens
                temperature=0.8,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and print
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"🤖 Generated: {generated_text}")

if __name__ == "__main__":
    main()