from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

# chat = ChatOpenAI(temperature=0)

chat = ChatOpenAI(temperature=0, openai_api_key="...")


template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

def translator_lc(txt):
    # get a chat completion from the formatted messages
    contx = chat.invoke(
        chat_prompt.format_prompt(
            input_language="Chinese", output_language="Vietnamese", text=txt,
        ).to_messages()
    ).content
    return contx

