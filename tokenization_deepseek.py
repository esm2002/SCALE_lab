from transformers import LlamaTokenizerFast, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch #version 2.5.1
import argparse
import os

def main(args):
    device = 'cpu'

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} not found.")
    
    os.makedirs('./results', exist_ok=True)
    token_ids_path = f'./results/token_ids.txt'
    if os.path.isfile(token_ids_path): os.remove(token_ids_path)
    output_path = f'./results/output.txt'
    if os.path.isfile(output_path): os.remove(output_path)

    # Step 1: Tokenize the input file
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    #llama_tokenizer = LlamaTokenizerFast.from_pretrained(args.model_path)

    tokenized_data = []
    seen_lines = set()
    
    with open(args.input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if line and line not in seen_lines:
                seen_lines.add(line)
                token_ids = tokenizer.encode(line)
                tokenized_data.append(token_ids)

            #if line:
                #token_ids = tokenizer.encode(line) #tokenizer(line, return_tensors="pt", padding=True).to(device) 
                #tokenized_data.append(token_ids)
                #llama_token_ids = llama_tokenizer.encode(line)
                #print(llama_token_ids, "\n")
    
    # Step 2: Save token ids to the output file
    with open(token_ids_path, 'w') as f:
        for token_ids in tokenized_data:
            # Convert token ids to string format and save to output file
            f.write(" ".join(map(str, token_ids)) + "\n")

    # Step 3: Run a LLM model    
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()
    model.to(device)

    outputs = []
    
    for token_ids in tokenized_data:
        input_ids = torch.tensor([token_ids]) #token_ids["input_ids"] 
        attention_mask = torch.ones(input_ids.shape, device=device) #token_ids["attention_mask"] 
        pad_token_id = tokenizer.pad_token_id 
        eos_token_id = tokenizer.eos_token_id 
        
        with torch.no_grad():
            results = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                do_sample=True, 
                max_length=50, 
                top_k=20, 
                top_p=0.95, 
                num_return_sequences=1,
                repetition_penalty = 2.0
            ) # encoded answer 
        outputs.append(tokenizer.batch_decode(results, skip_special_tokens=True)[0])
            #output_text = tokenizer.batch_decode(results, skip_special_tokens=True)[0]
            #clean_output = "".join([char for char in output_text if char.isprintable()])
            #outputs.append(clean_output)

            
    # Step 4: Save results to the output file
    with open(output_path, 'w') as f:
        f.write("<input and output>\n")
        for o in outputs:
            f.write(f'{o}\n')

    # Command to run:
    # python3 tokenization_deepseek.py test_input.txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and extract embeddings.")
    parser.add_argument("input_file", type=str, help="Path to the input text file.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B") #"deepseek-ai/DeepSeek-R1"
    
    args = parser.parse_args()
    main(args)
    
