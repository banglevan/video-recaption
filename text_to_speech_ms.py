import edge_tts
import os
import asyncio

TEXT = ('The sunlight sparkled on the waves, '
        'making the entire beach glisten and shimmer. '
        'I lay on the lounge chair, my eyes closed, '
        'relishing the gentle sensation of the sea breeze caressing my bronzed skin') #mock-up

VOICE = 'en-US-SteffanNeural'

ROOT_TO_SAVE = 'outputs'
TOPIC_CODES = 'topic_1'

MP3_FILE = 'example_text_to_speech.mp3'
VTT_FILE = 'example_text_to_speech.vtt'

PATH_TO_SAVE = os.path.join(ROOT_TO_SAVE, TOPIC_CODES)
if not os.path.exists(PATH_TO_SAVE):
    os.makedirs(PATH_TO_SAVE)

OUTPUT_FILE = os.path.join(PATH_TO_SAVE, MP3_FILE)
WEBVTT_FILE = os.path.join(PATH_TO_SAVE, VTT_FILE)
async def amain() -> None:
    global TEXT, VOICE, OUTPUT_FILE, WEBVTT_FILE
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE, rate='-20%')
    submaker = edge_tts.SubMaker()
    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    with open(WEBVTT_FILE, "w", encoding="utf-8") as file:
        file.write(submaker.generate_subs())

loop = asyncio.get_event_loop_policy().get_event_loop()

try:
    loop.run_until_complete(amain())
except:
    pass
