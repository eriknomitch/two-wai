import os
import gradio as gr
import whisper
from whisper import tokenizer
import time

current_size = 'base'
model = whisper.load_model(current_size)

def transcribe(audio, state={}, model_size='base', delay=1.2):
    time.sleep(delay - 1)

    global current_size
    global model
    if model_size != current_size:
        current_size = model_size
        model = whisper.load_model(current_size)

    transcription = model.transcribe(
        audio,
        language='english'
    )
    state['transcription'] += transcription['text'] + " "

    return state['transcription'],  state


title = "two-way-speech"
description = "A demo of two-way-speech"

model_size = gr.Dropdown(label="Model size", choices=['base', 'tiny', 'small', 'medium', 'large'], value='tiny')

delay_slider = gr.inputs.Slider(minimum=1, maximum=5, default=1.2, label="Rate of transcription")

transcription_tb = gr.Textbox(label="Transcription", lines=10, max_lines=200)

state = gr.State({"transcription": ""})

gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="microphone", type="filepath", streaming=True),
        state,
        model_size,
        delay_slider,
        ],
    outputs=[
        transcription_tb,
        state
    ],
    live=True,
    allow_flagging='never',
    title=title,
    description=description,
).launch(
    # enable_queue=True,
    debug=True,
    listen=True
  )
