from app.tokenizer import BPETokenizer
from app.cli import build_parser

def test_tokenizer_round_trip_known_text():
    tok = BPETokenizer().train(["rice needs water", "wheat needs care", "rice grows well"], vocab_size=80)
    ids = tok.encode("rice needs water")
    decoded = tok.decode(ids)
    assert "rice" in decoded
    assert "water" in decoded

def test_cli_parser_accepts_advice_command():
    args = build_parser().parse_args(["advise", "--crop", "rice", "--problem", "yellow leaves"])
    assert args.command == "advise"
    assert args.crop == "rice"
