from transformers import pipeline
from transformer_bert import compute_bert_score
from test_transformer import get_google_translation

def main(text):
  translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
  # text = "一九七八年了中国开始既苹开放，积极学习西方严业楼式了同时化为已用了将其亮分本士化，这种楼式并不完全照搬西方个而是结合方中国的文化知知识背景，航戚了猫特的申国特色社会主火。"
  result = translator(text)
  pred = result[0]['translation_text']
  print(pred)

  google = get_google_translation(text)
  bert = compute_bert_score(pred, google)
  print(bert)

  bert_score = float(bert) if hasattr(bert, 'item') else bert
  return {"prediction": pred, "bert": bert_score}