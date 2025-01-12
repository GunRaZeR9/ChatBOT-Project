from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can choose a more advanced model if needed
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_story(input_prompt):
    # Encode the input prompt
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')

    # Generate output
    output = model.generate(input_ids, 
                            max_length=500,  # Adjust length as needed
                            num_return_sequences=1,
                            no_repeat_ngram_size=2, 
                            do_sample=True, 
                            top_k=50, 
                            top_p=0.95, 
                            temperature=1.0)

    # Decode the output
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

if __name__ == "__main__":
    print("Welcome to the Fantasy Story Generator!")
    
    while True:
        user_input = input("Enter a prompt for your fantasy story (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
            
        story = generate_story(user_input)
        print("\nGenerated Story:\n")
        print(story)
        print("\n" + "="*50 + "\n")