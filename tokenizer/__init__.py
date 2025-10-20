"""
Python wrapper for the Rust-based tokenizer.
This module provides a high-level API for tokenization tasks
compatible with the TensolR library.
"""

try:
    from .tensolr_tokenizer import Tokenizer as RustTokenizer
except ImportError:
    # Fallback if Rust extension is not built
    RustTokenizer = None
    print("Warning: Rust tokenizer not available. Please build the Rust extension.")


class Tokenizer:
    """
    High-level tokenizer API for NLP tasks.
    Wraps the Rust implementation with Python-friendly methods.
    """
    
    def __init__(self, vocab_size=30000):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Size of the vocabulary to use
        """
        if RustTokenizer is None:
            raise ImportError("Rust tokenizer module not found. Please build the Rust extension.")
        
        self._rust_tokenizer = RustTokenizer()
        self.vocab_size = vocab_size
        self._is_trained = False
    
    def train(self, texts, special_tokens=None):
        """
        Train the tokenizer on a list of texts.
        
        Args:
            texts: List of strings to train on
            special_tokens: List of special tokens to include in vocabulary
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        
        self._rust_tokenizer.train(texts, self.vocab_size, special_tokens)
        self._is_trained = True
    
    def encode(self, text):
        """
        Encode a text string into token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        return self._rust_tokenizer.encode(text)
    
    def decode(self, token_ids):
        """
        Decode a list of token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not self._is_trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        return self._rust_tokenizer.decode(token_ids)
    
    def save(self, path):
        """
        Save the tokenizer to a file.
        
        Args:
            path: Path to save the tokenizer
        """
        return self._rust_tokenizer.save(path)
    
    def load(self, path):
        """
        Load the tokenizer from a file.
        
        Args:
            path: Path to load the tokenizer from
        """
        result = self._rust_tokenizer.load(path)
        self._is_trained = True
        return result


def create_basic_tokenizer(vocab_size=10000):
    """
    Create a basic tokenizer instance with default settings.
    
    Args:
        vocab_size: Size of the vocabulary
        
    Returns:
        Tokenizer instance
    """
    return Tokenizer(vocab_size=vocab_size)


# Example usage
if __name__ == "__main__":
    # Example of how to use the tokenizer
    texts = [
        "Hello world!",
        "How are you?",
        "I am fine, thank you.",
        "This is a sample text for training."
    ]
    
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.train(texts)
    
    # Test encoding and decoding
    test_text = "Hello world"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")