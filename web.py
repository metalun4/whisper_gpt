import gradio as gr
import openai
import config

openai.api_key = config.OPENAI_KEY
prev_messages = [{"role": "system", "content": "You are a ChatHR. You are assigned to do employee surveys and analyze them."}]


def respond(audio):
    global prev_messages
    audio_file = open(audio, 'rb')
    transcript = openai.Audio.transcribe('whisper-1', audio_file)

    prev_messages.append({'role': 'user', 'content': transcript['text']})

    response = openai.ChatCompletion.create(model="gpt-4", messages=prev_messages)

    gpt_message = response['choices'][0]['message']['content']
    prev_messages.append({'role': 'assistant', 'content': gpt_message})

    chat_transcript = ''

    for message in prev_messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ': ' + message['content'] + '\n\n'

    return chat_transcript


ui = gr.Interface(fn=respond, inputs=gr.Audio(source='microphone', type='filepath'), outputs='text')

ui.launch()
