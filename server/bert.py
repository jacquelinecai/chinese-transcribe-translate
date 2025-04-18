from bert_score import score

# Example sentences (predicted and reference)
predicted_english = [
    "In 1978, China began to open up and actively learned from Western business models and localized them. This model was not completely copied from the West, but combined with China's cultural and knowledge background to create a unique Chinese socialist system."
]

reference_english = [
    "In 1978, China began its reform and opening up, actively learning from Western industrial models and adapting them to its own needs. This model was not a complete copy of the West, but was combined with China's cultural and knowledge background to form a unique socialism with Chinese characteristics."
]

# Compute BERTScore
P, R, F1 = score(predicted_english, reference_english, lang="en")

# Print the results
print(f"Precision: {P[0]:.4f}")
print(f"Recall: {R[0]:.4f}")
print(f"F1 Score: {F1[0]:.4f}")