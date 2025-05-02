# ------------------------------------------------------------------
# Dependencies:
#   pip install torch torchvision transformers googletrans==3.1.0a0 scikit-learn
# ------------------------------------------------------------------
import os
import torch
import torch.nn.functional as F
from googletrans import Translator
from transformer import Transformer, TranslationDataset
from transformer_bert import compute_bert_score

def translate_with_model(model, device, sentence: str) -> str:
    """Tokenize Chinese properly - character by character"""
    # Build vocab from training dataset to match checkpoint
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_json = os.path.join(base_dir, '..', 'data', 'translation2019zh', 'translation2019zh_train.json')
    train_ds = TranslationDataset(train_json)
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab
    id2tgt = {i: tok for tok, i in tgt_vocab.items()}
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    # Tokenize Chinese character by character (this is the key fix)
    tokens = list(sentence.lower())  # Split into individual characters
    print(f"Source tokens: {tokens}")
    
    src_ids = [src_vocab['<sos>']] + [src_vocab.get(w, src_vocab['<unk>']) for w in tokens] + [src_vocab['<eos>']]
    print(f"Source IDs: {src_ids}")
    
    # Check how many tokens are unknown
    unknown_count = src_ids.count(src_vocab['<unk>'])
    print(f"Unknown tokens: {unknown_count}/{len(src_ids)}")
    
    src_tensor = torch.tensor([src_ids]).to(device)

    # Greedy decode with debugging
    sos, eos = tgt_vocab['<sos>'], tgt_vocab['<eos>']
    tgt_ids = [sos]
    max_len = 100
    
    print("\n--- Decoding Process ---")
    for i in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids]).to(device)
        with torch.no_grad():
            out = model(src_tensor, tgt_tensor)  # shape [1, seq_len, vocab_size]
            
        # Get top 3 predictions for the last token
        probs, indices = torch.topk(out[0, -1], 3)
        next_id = int(indices[0])
        
        print(f"Step {i}: Top 3 predictions: {[(id2tgt.get(int(idx), '<unk>'), float(prob)) for idx, prob in zip(indices, probs)]}")
        
        tgt_ids.append(next_id)
        if next_id == eos:
            break
    
    # Convert IDs to tokens
    pred_tokens = [id2tgt[i] for i in tgt_ids[1:] if i in id2tgt and i != eos]
    print(f"Final output IDs: {tgt_ids}")
    print(f"Final output tokens: {pred_tokens}")
    
    return ' '.join(pred_tokens)


def get_google_translation(text, src='auto', dest='en'):
    """Get translation from Google Translate API synchronously"""
    try:
        translator = Translator()
        result = translator.translate(text, src=src, dest=dest)
        return result.text
    except Exception as e:
        print(f"Google Translation Error: {e}")
        return f"[Translation Error: {e}]"


def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data', 'translation2019zh')
    model_path = os.path.join(base_dir, 'transformer_lr_fixed_final.pth')

    # Print debugging info
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Model path: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")

    # Load training vocab & instantiate model
    train_json = os.path.join(data_dir, 'translation2019zh_train.json')
    print(f"Training data path: {train_json}")
    print(f"Training data exists: {os.path.exists(train_json)}")
    
    try:
        train_ds = TranslationDataset(train_json)
        print("Successfully loaded training dataset")
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        return
    
    # Update model parameters to match the checkpoint
    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=256,
        num_heads=4,
        num_layers=3,
        d_ff=1024,
        max_seq_length=256,
        dropout=0.35
    )
    print("Successfully created model")
    print(f"Model architecture:\n{model}")

    # Load weights with error handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model_weights = torch.load(model_path, map_location=device)
        print("Successfully loaded model weights")
        print(f"Number of parameters in checkpoint: {len(model_weights)}")

        model_weights = {k: v.float() for k, v in model_weights.items()}
        
        # Print first few keys to check structure
        print("First 5 keys in checkpoint:")
        for i, key in enumerate(list(model_weights.keys())[:5]):
            print(f"{key}: {model_weights[key].shape}")
        
        model.load_state_dict(model_weights)
        print("Successfully loaded weights into model")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    model.to(device)
    model.eval()

    # Test a simple case first
    simple_test = "你好"  # "Hello" in Chinese
    print("\n=== Testing with simple input: '你好' ===")
    translate_with_model(model, device, simple_test)

    # Test sentences
    test_sentences = [
        "这次行程没计划好,其实可以多去九个地方逛逛一重读热闹的街区我们也没去一有点小造熊。例是吃了顿老火铅一只是感觉都是口水一没有尽兴。"
    ]

    all_preds, all_refs = [], []

    for sent in test_sentences:
        print(f"\n=== Testing with full input ===")
        # Model translation
        pred = translate_with_model(model, device, sent)
        
        # Google reference - try with error handling
        print("\nGetting Google translation...")
        ref = get_google_translation(sent)

        print(f"\nSource : {sent}")
        print(f"Model  : {pred}")
        print(f"Google : {ref}\n")

        all_preds.append(pred)
        all_refs.append(ref)

    # Only compute BERT-score if we have predictions and references
    if all_preds and all_refs:
        score = compute_bert_score(all_preds, all_refs)
        print(f"Mean BERT-score: {score:.4f}")
    else:
        print("No translations completed successfully. Cannot compute BERT-score.")


if __name__ == "__main__":
    main()