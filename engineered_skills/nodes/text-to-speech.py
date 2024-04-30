#!/usr/bin/env python3

import rospy
import pyaudio
import wave

from engineered_skills.srv import Speech,SpeechResponse

RUNNING = -1
SUCCESS = 1
NOT_STARTED = 2

class SpeechPlanner:
    def __init__(self):
        rospy.init_node("hal_speech_node")
        self.node_name = rospy.get_name()
        rospy.loginfo("{0} started".format(self.node_name))


    def start(self):

        self.action_status = NOT_STARTED

        s = rospy.Service('hal_speech_server', Speech, self.callback)
        rospy.loginfo("Speech server has started")
        rospy.spin()


    def main(self, speech_text):
        synthesize_text(speech_text)
        play_audio()


    def callback(self, req):
        print('callback called')
        if self.action_status == NOT_STARTED:
            self.action_status = RUNNING
            self.main(req.speech_text)
            self.action_status = NOT_STARTED
            return SpeechResponse(SUCCESS)
        return SpeechResponse(self.action_status)



def get_respeaker_device_id():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    device_id = -1
    for i in range(num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            if "ReSpeaker" in p.get_device_info_by_host_api_device_index(0, i).get('name'):
                device_id = i

    return device_id

# RESPEAKER_RATE = 16000
RESPEAKER_RATE = 25000
RESPEAKER_CHANNELS = 6
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = get_respeaker_device_id()
CHUNK = 1024


def synthesize_text(input_text):
    """Synthesizes speech from the input string of text."""
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=input_text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-D",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # The response's audio_content is binary.
    with open("output.wav", "wb") as out:
        # out.setchannels(2)
        # out.setsampwidth(2)
        # out.setframerate(RESPEAKER_RATE)
        out.write(response.audio_content)
        print('Audio content written to file "output.wav"')


def play_audio():
    p = pyaudio.PyAudio()
    stream = p.open(rate=RESPEAKER_RATE,
                    format=p.get_format_from_width(RESPEAKER_WIDTH),
                    channels=1,
                    output=True)

    # for f in frames:
    #     stream.write(f, CHUNK)

    with wave.open("output.wav", 'rb') as fh:
        data = fh.readframes(CHUNK)
        while data != b'':
            stream.write(data)
            data = fh.readframes(CHUNK)
            

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    speech_planner = SpeechPlanner()

    speech_planner.start()
