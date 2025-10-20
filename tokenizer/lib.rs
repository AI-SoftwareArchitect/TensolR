use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::collections::HashSet;
use std::cmp::Ordering;

#[pyclass]
#[derive(Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    vocab_r: HashMap<usize, String>,
    merges: HashMap<(String, String), (String, usize)>,
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
            let mut next_id = 0;
            for token in &special_tokens_vec {
                self.special_tokens.insert(token.clone(), next_id);
                self.vocab.insert(token.clone(), next_id);
                self.vocab_r.insert(next_id, token.clone());
                next_id += 1;
            }
        }

        // Collect all unique words and their frequencies
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for text in &texts {
            for word in text.split_whitespace() {
                // Normalize and clean the word
                let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if !clean_word.is_empty() {
                    *word_freqs.entry(clean_word).or_insert(0) += 1;
                }
            }
        }

        // Create initial vocabulary with individual characters
        let mut chars: HashSet<char> = HashSet::new();
        for word in word_freqs.keys() {
            for c in word.chars() {
                chars.insert(c);
            }
        }

        // Add character tokens to vocabulary
        let mut next_id = self.special_tokens.len();
        for c in chars {
            let char_str = c.to_string();
            if !self.vocab.contains_key(&char_str) {
                self.vocab.insert(char_str.clone(), next_id);
                self.vocab_r.insert(next_id, char_str);
                next_id += 1;
            }
        }

        // Perform BPE merges up to vocab_size
        let mut vocab_size_current = next_id;
        while vocab_size_current < vocab_size {
            let most_frequent_pair = self.get_most_frequent_pair(&word_freqs);
            
            if let Some((first, second, freq)) = most_frequent_pair {
                let new_token = format!("{}{}", first, second);
                
                // Add the new token to vocabulary
                self.vocab.insert(new_token.clone(), vocab_size_current);
                self.vocab_r.insert(vocab_size_current, new_token.clone());
                
                // Record the merge operation
                self.merges.insert((first.clone(), second.clone()), (new_token.clone(), vocab_size_current));
                
                // Update word frequencies by merging pairs
                self.update_word_freqs(&mut word_freqs, &first, &second);
                
                vocab_size_current += 1;
            } else {
                // No more pairs to merge
                break;
            }
        }
    }

    fn encode(&self, text: String) -> Vec<usize> {
        // Simple whitespace tokenization for now
        // In a full implementation, we would apply BPE merges
        let mut tokens = Vec::new();
        
        for word in text.split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            
            if let Some(&id) = self.special_tokens.get(&clean_word) {
                tokens.push(id);
                continue;
            }
            
            // For now, just split into characters if not in vocab
            if let Some(&id) = self.vocab.get(&clean_word) {
                tokens.push(id);
            } else {
                // Try to encode as character sequence
                let mut word_tokens = Vec::new();
                for c in clean_word.chars() {
                    let char_str = c.to_string();
                    if let Some(&id) = self.vocab.get(&char_str) {
                        word_tokens.push(id);
                    } else {
                        // Use unknown token if available
                        if let Some(&unk_id) = self.special_tokens.get("<unk>") {
                            word_tokens.push(unk_id);
                        } else {
                            word_tokens.push(0); // Default unknown token
                        }
                        break;
                    }
                }
                tokens.extend(word_tokens);
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

impl Tokenizer {
    fn get_most_frequent_pair(&self, word_freqs: &HashMap<String, usize>) -> Option<(String, String, usize)> {
        let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
        
        for (word, &freq) in word_freqs {
            let chars: Vec<char> = word.chars().collect();
            for i in 0..chars.len() - 1 {
                let pair = (chars[i].to_string(), chars[i+1].to_string());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }
        
        pair_freqs
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1))
            .map(|(pair, freq)| (pair.0, pair.1, freq))
    }
    
    fn update_word_freqs(&self, word_freqs: &mut HashMap<String, usize>, first: &str, second: &str) {
        let new_word = format!("{}{}", first, second);
        
        // Create a temporary new dictionary to avoid borrowing issues
        let mut new_word_freqs = HashMap::new();
        for (word, &freq) in word_freqs.iter() {
            if word.contains(&new_word) {
                new_word_freqs.insert(word.clone(), freq);
            } else {
                new_word_freqs.insert(word.clone(), *freq);
            }
        }
        
        // Update the original dictionary
        *word_freqs = new_word_freqs;
    }
}

// Python module
#[pymodule]
fn tensolr_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}