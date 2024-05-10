from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


article_en = "我们非常开心由于您的到来"
model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate").to("cuda")
tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="zh_CN")

model_inputs = tokenizer(article_en, return_tensors="pt").to("cuda")

import time
for i in range(100):
    t1 = time.time()
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"]
    )
    x = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(x, time.time() - t1)
