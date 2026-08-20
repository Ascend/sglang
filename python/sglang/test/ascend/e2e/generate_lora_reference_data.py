import argparse
import os
os.environ["PEFT_NO_TORCHAO"] = "1"
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path",required=True,)
    parser.add_argument("--output",required=True,)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        # device="npu:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to("npu:0")

    print(f"Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()  # Merge LoRA into base model weights
    model = model.to("npu:0")
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("npu:0")

    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Compute logprobs
    logprobs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]

    # Get logprob for each actual token (shifted: token[i] -> logprob from logits[i-1])
    # logprobs[0, i-1, token_i] = log P(token_i | tokens[:i])
    token_ids = input_ids[0].tolist()
    token_logprobs = []
    for i in range(1, len(token_ids)):
        lp = logprobs[0, i - 1, token_ids[i]].item()
        token_logprobs.append(lp)

    # Save reference data
    data = {
        "tokens": token_ids,
        "training_logprobs": token_logprobs,
        "sampling_logprobs": token_logprobs,  # Same as training for reference
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved reference data to: {args.output}")
    print(f"Tokens: {len(token_ids)}")
    print(f"Logprobs: {len(token_logprobs)}")
    print(f"First few logprobs: {token_logprobs[:5]}")


if __name__ == "__main__":
    main()
