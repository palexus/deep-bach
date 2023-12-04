import music21 as m21
import numpy as np
import jsonlines
import os
from data_preprocessing import BACH_CHORALES, ACCEPTABLE_DURATIONS,\
                               has_acceptable_durations, transpose


SAVEDIR = "data_chorales_gpt3"
ONE_FILE_PATH = "choral_data_gpt3.txt"
SEQUENCE_LENGTH = 128


def encode_song(song, savedir, song_filename, time_step=0.25):
    """
        Converts a score into a time-series-like music representation. Each item in 
        the encoded list represents 'min_duration' quarter lengths. The symbols used at 
        each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
        for representing notes/rests that are carried over into a new time step. 
        It saves the song automatically to a file.

        In comparison to encode_song from data_preprocessing, it stores top to bottom and
        not from left to right. Since we use the language model we dont need numbers.


        PARAMETERS:
        -------------
        song :              Piece to encode (m21 stream)
        savedir :           Directory where the encoded song should be saved
        song_filename :     Name of file
        time_step :         Duration of each time step in quarter length

    """
    encoded_song = []
    for part in song.parts:
        encoded_part = []
        for event in part.flat.notesAndRests:            
            steps = ["_"] * (int(event.duration.quarterLength / time_step) - 1)
            if isinstance(event, m21.note.Note):
                symbol = str(event.pitch)
            if isinstance(event, m21.note.Rest):
                symbol = "r"
            
            encoded_part.append(symbol)
            encoded_part.extend(steps)
        encoded_song.append(encoded_part)

    transposed = np.array(encoded_song).T.tolist()
    encoded_text = "\n".join([" ".join(map(str, part)) for part in transposed])

    with open(os.path.join(savedir, song_filename), "w") as fp:
        fp.write(encoded_text)


def create_single_file(data_path, sequence_length):
    """
        Creates a single file out of all files in a directory and
        set a delimiter at the end of each piece. Here we have
        four voices which are concatenated together with the 
        delimiter in between. The delimiter is a sequence of length
        sequence_length such that later sequences can be read off
        easily. It also writes everything to a txt file.

        PARAMETERS:
        ---------------
        data_path :         path to directory
        sequence_length :   length of the sequences fed into the model

        RETURNS:
        ----------------
        songs :     the string object of all four tracks and all songs

    """
    delimiter = "/ / / /\n" * sequence_length
    with open(ONE_FILE_PATH, "w") as fp:
        fp.write("")

    for path, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path, "r") as fp:
                song = fp.read()
            with open(ONE_FILE_PATH, "a") as fp:
                fp.write(song)
                fp.write("\n")
                fp.write(delimiter)    


def preprocess(songs, durations, save_dir):
    """
        Creates a directory and saves all preprocessed songs
        into that folder. The songs are checked if they have 
        notes of acceptable length since we only consider 
        notes which match to a 16th note grid.
        All pieces are transposed to C Major or A Minor.

        PARAMETERS:
        ----------------
        songs :     songs to preprocess. In this case the Bach chorales
        durations:  acceptable durations
        save_dir :  path to directory (created only if not exists)
        
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, durations):
            continue
        song = transpose(song)
        encode_song(song, save_dir, str(i)+".txt")


def generate_training_sequences_empty(data_path):
    jsons = []
    for path, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path, "r") as fp:
                song = fp.read()
            jsons.append({"prompt": "", "completion": song + "\nEND"})
    with jsonlines.open('empty_prompt_data.jsonl', mode='w') as writer:
        writer.write_all(jsons)




if __name__=="__main__":
    print("Start preprocessing")
    #preprocess(BACH_CHORALES, ACCEPTABLE_DURATIONS, SAVEDIR)
    #create_single_file(SAVEDIR, SEQUENCE_LENGTH)
    generate_training_sequences_empty(SAVEDIR)
    print("Finished")