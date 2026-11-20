"""Command-line interface for training and requesting crop advice."""
import argparse
from pathlib import Path
from .storage import load_corpus, load_checkpoint
from .llm import train_model, generate_advice, choose_device

DISCLAIMER = (
    "Educational drafting aid only. Verify recommendations with local agricultural "
    "extension services or a qualified agronomist. Do not use as the sole basis for "
    "pesticide, fertilizer, irrigation, or food-safety decisions."
)

def build_parser():
    parser = argparse.ArgumentParser(prog="crop-advice", description="Generate short crop-care advice snippets.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train the small GPT model on a text corpus.")
    train.add_argument("--corpus", default="data/agriculture_corpus.txt")
    train.add_argument("--output", default="checkpoints/crop_gpt.pt")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--block-size", type=int, default=96)
    train.add_argument("--vocab-size", type=int, default=400)
    train.add_argument("--device", default="auto")
    ask = sub.add_parser("advise", help="Generate an advice snippet from a trained model.")
    ask.add_argument("--model", default="checkpoints/crop_gpt.pt")
    ask.add_argument("--crop", required=True)
    ask.add_argument("--problem", default="general crop care")
    ask.add_argument("--max-new-tokens", type=int, default=70)
    ask.add_argument("--temperature", type=float, default=0.7)
    ask.add_argument("--device", default="auto")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    print(DISCLAIMER)
    if args.command == "train":
        texts = load_corpus(args.corpus)
        train_model(texts, args.output, steps=args.steps, batch_size=args.batch_size,
                    block_size=args.block_size, vocab_size=args.vocab_size, device=args.device)
        print(f"\nModel saved to {args.output}")
    else:
        device = choose_device(args.device)
        model, tokenizer, _ = load_checkpoint(args.model, device)
        prompt = f"Crop: {args.crop}. Problem: {args.problem}. Advice:"
        answer = generate_advice(model, tokenizer, prompt, args.max_new_tokens,
                                 args.temperature, device)
        print(f"\n{answer or 'The model did not generate a continuation. Try a different prompt or train longer.'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
