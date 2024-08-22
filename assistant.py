from groq import Groq
from PIL import ImageGrab, Image
import cv2
import pyperclip
import google.generativeai as genai
import os
from faster_whisper import WhisperModel
import speech_recognition as sr
import time
import re
from openai import OpenAI
import pyaudio
import asyncio
import threading
import sched
from datetime import datetime, timedelta
import webbrowser
import openai


wake_word = 'jarvis'

# ENTER YOUR API KEYS HERE:
groq_client = Groq(api_key="https://console.groq.com/keys")
genai.configure(api_key="https://ai.google.dev")
openai_client = OpenAI(api_key="https://platform.openai.com/api-keys")
openai.api_key = "https://platform.openai.com/api-keys"

# WARNING, if webcam is not being captured change this value to 0,1,2,..
web_cam = cv2.VideoCapture(0)

sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already processed into a highly detailed '
    'text prompt that will be attached to their transcribe voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
    'You can also set alarms based on the user’s requests.'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': "BLOCK_NONE"
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

r = sr.Recognizer()
source = sr.Microphone()


def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n     IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content. '
        'taking a screnshot, capturing the webcam or calling no functions is best for a voice assistant to response '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'response with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "search web", "set reminder/alarm", "None"] \n'
        'Do not response with anything but the most logical selection from that list with no explanatations. Do not capture webcam unless there is '
        'EXPLICIT consent given. Format the function call name exactly as I listed.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message

    return response.content


# set alarm/reminder functions:
scheduler = sched.scheduler(time.time, time.sleep)

def set_alarm(alarm_info):
    def alarm_action():
        print('Alarm ringing!')
        speak('Alarm ringing!')

    alarm_time_str = alarm_info[1].rstrip(".")

    alarm_time = datetime.strptime(alarm_time_str, "%Y-%m-%d %H:%M:%S") 
    alarm_timestamp = time.mktime(alarm_time.timetuple())
    scheduler.enterabs(alarm_timestamp, 1, alarm_action)
    threading.Thread(target=scheduler.run).start()

    return f'Alarm set for {alarm_time.strftime("%Y-%m-%d %H:%M:%S")}'



def set_reminder(reminder_info):
    def reminder_action():
        print(f'Reminder: {reminder_info[2]}')
        speak(f'Reminder: {reminder_info[2]}')

    reminder_time_str = reminder_info[1].rstrip(".")
    reminder_time = datetime.strptime(reminder_time_str, "%Y-%m-%d %H:%M:%S")  # CHANGE MADE: Parse the date-time string
    reminder_timestamp = time.mktime(reminder_time.timetuple())
    scheduler.enterabs(reminder_timestamp, 1, reminder_action)
    threading.Thread(target=scheduler.run).start()

    return f'Reminder set for {reminder_time.strftime("%Y-%m-%d %H:%M:%S")}: {reminder_info[2]}'



def check_alarm_reminder(prompt):
    now = datetime.now()
    response = openai.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"The current time is {now}, Given the following user input prompt, extract the information from it and output it in the following format: \
                 REMINDER/ALARM, yyyy-mm-dd hh:mm:ss, [reminder message ONLY IF IT IS A REMINDER]. \
                \
                 explanation: \
                 insert REMINDER or ALARM in the first value depending on the prompt, if the user's prompt is inquiring for an alarm insert ALARM and vice versa, \
                 the second input will be the time, and the third input will be the message for the reminder IF AND ONLY IF the first index is REMINDER, this is where you will insert what the  \
                 user wants to be reminded for."},
                {"role": "user", "content": prompt}
            ]
        )
    result = response.choices[0].message.content
    result = [r.strip() for r in result.split(",")]

    if result[0] == "REMINDER":
        set_reminder(result)
    elif result[0] == "ALARM":
        set_alarm(result)
    else:
        return "Sorry, I couldn't understand your request."




def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)

def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Camera unavailable.')
        exit()
    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No clipboard text to copy")
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semtantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

def search_web(query):
    search_url = f"https://www.google.com/search?q={query}"
    webbrowser.open(search_url)
    return f"Searching the web for {query}"

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    with openai_client.audio.speech.with_streaming_response.create(
        model = 'tts-1',
        voice='alloy',
        response_format='pcm',
        input=text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

async def process_audio(prompt_text):
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        print(f'USER: {clean_prompt}')

        call = function_call(clean_prompt)
        if 'set reminder/alarm' in call:
            alarm_reminder_response = check_alarm_reminder(clean_prompt)
            if alarm_reminder_response:
                print(f'ASSISTANT: {alarm_reminder_response}')
                await asyncio.to_thread(speak, alarm_reminder_response)
                return
        if 'take screenshot' in call:
            print('Taking screenshot')
            await asyncio.to_thread(take_screenshot)
            visual_context = await asyncio.to_thread(vision_prompt, prompt=clean_prompt, photo_path='screenshot.jpg')
            os.remove('screenshot.jpg')
        elif 'capture webcam' in call:
            print('Capturing webcam')
            await asyncio.to_thread(web_cam_capture)
            visual_context = await asyncio.to_thread(vision_prompt, prompt=clean_prompt, photo_path='webcam.jpg')
            os.remove('webcam.jpg')
        elif 'extract clipboard' in call:
            print('Copying clipboard text')
            paste = await asyncio.to_thread(get_clipboard_text)
            clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
            visual_context = None
        elif "search web" in call:
            search_query = clean_prompt.replace("search the web for", "").strip()
            search_response = search_web(search_query)
            print(f'ASSISTANT: {search_response}')
            await asyncio.to_thread(speak, search_response)
            return
        
        else:
            visual_context = None

        response = await asyncio.to_thread(groq_prompt, prompt=clean_prompt, img_context=visual_context)
        print(f'ASSISTANT: {response}')
        await asyncio.to_thread(speak, response)

        os.remove('prompt.wav')

        if os.path.exists('response.mp3'):
            os.remove('response.mp3')

async def callback(recognizer, audio):
    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        
        prompt_text = wav_to_text(prompt_audio_path)
        await process_audio(prompt_text)
    except Exception as e:
        print(f"Error in callback: {str(e)}")

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print(f'\nSay "{wake_word}" followed by your prompt.\n')

    stop_listening = r.listen_in_background(source, lambda r, a: asyncio.run(callback(r, a)))
    
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        print("\nListening stopped.")

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None


start_listening()

