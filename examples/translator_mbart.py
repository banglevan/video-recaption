from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
article_zh = "专家在第六届联合国环境大会上赞扬中国在绿色出行方面的作用"
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to('cuda')
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# # translate Hindi to French
# tokenizer.src_lang = "hi_IN"
# encoded_hi = tokenizer(article_hi, return_tensors="pt")
# generated_tokens = model.generate(
#     **encoded_hi,
#     forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
# )
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# translate Arabic to English
tokenizer.src_lang = "zh_CN"
encoded_ar = tokenizer(article_zh, return_tensors="pt").to('cuda')
generated_tokens = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["sw_KE"]
)
x = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(x)
# => "The Secretary-General of the United Nations says there is no military solution in Syria."
