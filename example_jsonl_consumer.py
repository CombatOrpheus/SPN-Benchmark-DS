import json
import zstandard as zstd
import gzip
import lz4.frame
import io
import argparse

def consume_jsonl(file_path):
    """
    Demonstrates how to read and decompress a compressed JSONL file on the fly.
    This structure reads line by line without loading the entire uncompressed file into memory.
    """
    if file_path.endswith('.zst'):
        print(f"Reading ZStandard compressed file: {file_path}")
        dctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                process_stream(text_stream)

    elif file_path.endswith('.gz'):
        print(f"Reading Gzip compressed file: {file_path}")
        with gzip.open(file_path, "rt", encoding="utf-8") as text_stream:
            process_stream(text_stream)

    elif file_path.endswith('.lz4'):
        print(f"Reading LZ4 compressed file: {file_path}")
        with lz4.frame.open(file_path, "rt", encoding="utf-8") as text_stream:
            process_stream(text_stream)

    else:
        print(f"Reading uncompressed file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as text_stream:
            process_stream(text_stream)

def process_stream(text_stream):
    """
    Reads the text stream line by line and parses JSON.
    """
    try:
        first_line = next(text_stream)
        config = json.loads(first_line)
        print("Successfully read configuration block.")
    except StopIteration:
        print("File is empty.")
        return

    sample_count = 0
    for line in text_stream:
        sample = json.loads(line)
        sample_count += 1

        if sample_count == 1:
            print(f"\nFirst sample keys: {list(sample.keys())}")

    print(f"\nSuccessfully decompressed and read {sample_count} samples on the fly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume compressed JSONL output")
    parser.add_argument("file_path", type=str, help="Path to the JSONL file to read (.jsonl, .jsonl.zst, etc.)")
    args = parser.parse_args()
    consume_jsonl(args.file_path)
