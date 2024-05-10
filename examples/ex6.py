import openai

# Set up your API key
openai.api_key = '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'
# Define your prompt
prompt = "Translate the following Chinese text to vietnamese: 我们非常开心由于您的到来"

# Call the completion endpoint
response = openai.Completion.create(
  engine="gpt-3.5-turbo-instruct",  # You can choose a different engine as per your preference
  prompt=prompt,
  max_tokens=50,  # Adjust this parameter based on the length of the desired translation
  temperature=0.1,  # You can adjust the temperature parameter for diversity in the response
)

# Print the generated translation
for i in range(100):
  print(response.choices[0].text.strip())