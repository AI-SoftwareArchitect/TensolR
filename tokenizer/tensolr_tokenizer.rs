use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::collections::HashSet;

#[pyclass]
#[derive(Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    vocab_r: HashMap<usize, String>,
    merges: HashMap<(String, String), String>,
    special_tokens: HashMap<String, usize>,
    pattern: Regex,
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            merges: HashMap::new(),
            special_tokens: HashMap::new(),
            pattern: Regex::new(r"(\s+|[.,!?;:])|(\w+|\W)").unwrap(),
        }
    }

    fn train(&mut self, texts: Vec<String>, vocab_size: usize, special_tokens: Option<Vec<String>>) {
        // Initialize special tokens if provided
        if let Some(special_tokens_vec) = special_tokens {
            for (idx, token) in special_tokens_vec.iter().enumerate() {
                self.special_tokens.insert(token.clone(), idx);
            }
        }

        // Simple tokenization for training
        let mut pairs: HashMap<(String, String), usize> = HashMap::new();
        let mut words: HashMap<String, usize> = HashMap::new();

        // Tokenize and count all words in texts
        for text in texts {
            let tokens: Vec<&str> = text.split_whitespace().collect();
            for token in tokens {
                *words.entry(token.to_string()).or_insert(0) += 1;
            }
        }

        // Build initial vocabulary from most frequent tokens
        let mut sorted_words: Vec<_> = words.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));

        // Add special tokens to vocab
        for (token, id) in &self.special_tokens {
            self.vocab.insert(token.clone(), *id);
            self.vocab_r.insert(*id, token.clone());
        }

        // Add regular tokens to vocabulary up to vocab_size
        let mut next_id = self.special_tokens.len();
        for (word, _) in sorted_words.iter().take(vocab_size - self.special_tokens.len()) {
            self.vocab.insert(word.clone(), next_id);
            self.vocab_r.insert(next_id, word.clone());
            next_id += 1;
        }
    }

    fn encode(&self, text: String) -> Vec<usize> {
        let mut tokens = Vec::new();
        
        // Check if the whole text is a special token
        if self.special_tokens.contains_key(&text) {
            tokens.push(*self.special_tokens.get(&text).unwrap());
            return tokens;
        }

        // Split by whitespace and other separators
        for part in text.split_whitespace() {
            let token = part.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if !token.is_empty() {
                // Try to find in vocabulary
                if let Some(&id) = self.vocab.get(&token) {
                    tokens.push(id);
                } else {
                    // Handle out-of-vocabulary tokens - for now just use unknown token if available
                    let unknown_token_id = self.special_tokens.get("<unk>").unwrap_or(&0);
                    tokens.push(*unknown_token_id);
                }
            }
        }

        tokens
    }

    fn decode(&self, token_ids: Vec<usize>) -> String {
        let mut result = Vec::new();
        for id in token_ids {
            if let Some(token) = self.vocab_r.get(&id) {
                result.push(token.clone());
            } else if let Some(token) = self.special_tokens.iter().find(|(_, &val)| val == id).map(|(k, _)| k.clone()) {
                result.push(token);
            } else {
                result.push(format!("[UNK_ID:{}]", id));
            }
        }
        result.join(" ")
    }

    fn save(&self, path: String) -> PyResult<()> {
        // Simplified save implementation
        // In a real implementation, we would serialize the tokenizer data to a file
        println!("Saving tokenizer to: {}", path);
        Ok(())
    }

    fn load(&mut self, path: String) -> PyResult<()> {
        // Simplified load implementation
        // In a real implementation, we would deserialize the tokenizer data from a file
        println!("Loading tokenizer from: {}", path);
        Ok(())
    }
}

// Python module
#[pymodule]
fn tensolr_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}