import music21 as m21
import numpy as np
import base64
import os
import subprocess
import streamlit as st
from random import randint
from time import sleep

PATH_TO_CHORALES = "data_chorales_gpt3"


class Seed_data:
    def __init__(self):
        print("------One seed is created--------")
        self.choral = None
        self.path_pdf = None
        self.path_mid = None
        self.path_mp3 = None
        self.stream = None
    
    def gen_seed(self):
        return randint(0, 367)

    def displayPDF(self, file):
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


    def load_seed(self):
        seed = self.gen_seed()
        with open(os.path.join(PATH_TO_CHORALES, str(seed) + ".txt"), "r") as fp:
            choral = fp.read().split("\n")[:17]
            self.choral = "\n".join(choral)
        self.path_pdf, self.stream = save_piece(self.choral, step_duration=0.25, format="musicxml.pdf", file_name="midi_results/seed.pdf")
    

    def to_pdf(self):
        self.displayPDF(self.path_pdf)
    

    def create_midi(self):
        self.path_mid, _ = save_piece(self.choral, step_duration=0.25, format="midi", file_name="midi_results/seed.mid")
        sleep(3)

    def create_mp3(self):
        self.create_midi()
        self.path_mp3 = self.path_mid[:-3] + "mp3"
        subprocess.check_output(f"timidity {self.path_mid} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {self.path_mp3}", shell=True)


class Choral_data:
    def __init__(self, choral):
        self.choral = choral
        self.path_pdf = None
        self.path_mid = None
        self.path_mp3 = None
        self.stream = None


    def displayPDF(self, file):
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


    def to_pdf(self):
        self.path_pdf, self.stream = save_piece(self.choral, step_duration=0.25, format="musicxml.pdf", file_name="midi_results/output.pdf")
        self.displayPDF(self.path_pdf)
    

    def create_midi(self):
        self.path_mid, _ = save_piece(self.choral, step_duration=0.25, format="midi", file_name="midi_results/output.mid")


    def create_mp3(self):
        self.create_midi()
        self.path_mp3 = self.path_mid[:-3] + "mp3"
        subprocess.check_output(f"timidity {self.path_mid} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {self.path_mp3} >/dev/null", shell=True)


def first_n_bars(data, n):
    split_data = split_data = data.split("\n")
    return "\n".join(split_data[:n*16])


def last_n_bars(data, n):
    split_data = data.split("\n")
    return "\n".join(split_data[-n*16:])


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1

    return path


def save_piece(piece, step_duration=0.25, format="midi", file_name="mel.mid"):
    """
        Converts a piece into a MIDI file

        PARAMETERS:
        ---------------
        piece :             string representation of the piece
        step_duration :     for 16th node use 0.25
        format :            to which format do you want to convert the string representation, e.g. midi, musicxml
        file_name :         name of file

    """
    splitted = [m.split() for m in piece.split("\n")]

    if len(splitted[-1])!= len(splitted[0]):
        splitted = splitted[:-1]

    piece = np.array(splitted).T.tolist()

    # create a music21 stream
    stream = m21.stream.Stream()

    parts = [m21.stream.Part(id="Sopran"), m21.stream.Part(id="Alt"),
             m21.stream.Part(id="Tenor"), m21.stream.Part(id="Bass") ]

    
    step_counter = 1

    # parse all the symbols in the melody and create note/rest objects
    for j in range(4):

        start_symbol = None
        melody = piece[j]

        for i, symbol in enumerate(melody):

            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(start_symbol, quarterLength=quarter_length_duration)

                    parts[j].append(m21_event)
                    step_counter = 1

                start_symbol = symbol

            else:
                step_counter += 1

        stream.append(parts[j])

    file_name = uniquify(file_name)
    stream.write(format, file_name)

    return file_name, stream