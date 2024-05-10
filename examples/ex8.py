from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

# chat = ChatOpenAI(temperature=0)

chat = ChatOpenAI(temperature=0, openai_api_key="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")

# messages = [
#     SystemMessage(
#         content="You are a helpful assistant that translates English to French."
#     ),
#     HumanMessage(
#         content="Translate this sentence from English to French. I love programming."
#     ),
# ]
# chat.invoke(messages)


template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
x = chat.invoke(
    chat_prompt.format_prompt(
        input_language="Chinese", output_language="Vietnamese", text="我要跑的远远的"
    ).to_messages()
)

print(x.content)