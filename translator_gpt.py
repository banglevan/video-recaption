import openai

# Set up your API key
openai.api_key = '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'
# openai.api_key = "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"
# Define your prompt
# prompt = "Translate the following Chinese text to vietnamese: 我们非常开心由于您的到来"
sys_prompt = "Translate the following Chinese text to vietnamese: "
def openai_translator(user_promt):
    # Call the completion endpoint
    prompt = sys_prompt + user_promt
    response = openai.Completion.create(
      engine="gpt-3.5-turbo-instruct",  # You can choose a different engine as per your preference
      prompt=prompt,
      max_tokens=50,  # Adjust this parameter based on the length of the desired translation
      temperature=0.1,  # You can adjust the temperature parameter for diversity in the response
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    openai_translator('我们非常开心由于您的到来')