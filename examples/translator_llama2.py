from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate").to("cuda")
tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="zh_CN")

def inference(article):
    model_inputs = tokenizer(article, return_tensors="pt").to("cuda")
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated

if __name__ == '__main__':
    article = "我们非常开心由于您的到来"
    print(inference(article))