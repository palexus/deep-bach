"""
    This module is for data preprocessing. It loads the data, removes all Bach pieces which
    dont have four voices = only Bach chorales left.

    To perform well in all harmonies, we have two approaches. 
        * First is to enlarge the dataset by
          transposing and adding to it all pieces in all keys. This scales the dataset by 24.
        * Second option is to transpose every piece to C-major/A-minor such that the model 
          learns only to compose in these two harmonies. This makes the model faster and is
          better suited for the SPICED project.

"""
import music21 as m21
import numpy as np
import os
import pickle
import json
import tensorflow.keras as keras


with open("choral_data.bin", "rb") as file:
    print("Loading data")
    BACH_CHORALES = pickle.load(file)

ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
ENCODER_PATH = "encoder.json"
DECODER_PATH = "decoder.json"
SAVEDIR = "data_chorales"
ONE_FILE_PATH = "choral_data.txt"
SEQUENCE_LENGTH = 64


def has_acceptable_durations(song, acceptable_durations):
    """
        Boolean routine that returns True if piece has all acceptable duration.

        PARAMETERS
        ----------------
        song :                    Piece to check for durations as music21 stream
        acceptable_durations :    List of acceptable duration in quarter length

        RETURNS:
        ----------------
        boolean

    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """
        Transposes song to C maj/A min

        PARAMETERS:
        ----------------
        song :   Piece to transpose

        RETURNS:
        ----------------
        transposed song (as music21 stream)

    """
    try:
        key = song.parts[0].flat.getElementsByClass(m21.key.Key)[0]
    except:
        key = song.parts[1].flat.getElementsByClass(m21.key.Key)[0]     # lazy solution, improve later

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        print("There was no key")       # But choral data is nice, this shouldnt happen for the project
        key = song.analyze("key")

    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    tranposed_song = song.transpose(interval)

    return tranposed_song


def encode_song(song, savedir, song_filename, time_step=0.25):
    """
        Converts a score into a time-series-like music representation. Each item in 
        the encoded list represents 'min_duration' quarter lengths. The symbols used at 
        each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
        for representing notes/rests that are carried over into a new time step. 
        It saves the song automatically to a file.

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
                symbol = event.pitch.midi
            if isinstance(event, m21.note.Rest):
                symbol = "r"
            
            encoded_part.append(symbol)
            encoded_part.extend(steps)
        encoded_song.append(encoded_part)
    encoded_text = "\n".join([" ".join(map(str, part)) for part in encoded_song])

    with open(os.path.join(savedir, song_filename), "w") as fp:
        fp.write(encoded_text)


def create_dictionary(songs, encoder_path, decoder_path):
    """
        Creates a json file that maps the symbols in the song dataset onto integers

        PARAMETERS:
        ----------------
        songs :         String with all songs
        mapping_path :  Path where to save mapping
        
    """
    song = songs.split()
    vocabulary = sorted(set(song))

    encoder = {symbol:i for i, symbol in enumerate(vocabulary)}
    decoder = {i:symbol for i, symbol in enumerate(vocabulary)}

    with open(encoder_path, "w") as fp:
        json.dump(encoder, fp, indent=4)
    with open(decoder_path, "w") as fp:
        json.dump(decoder, fp, indent=4)
    
    return encoder, decoder
    

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
    delimiter = "/ " * sequence_length
    tracks = 4*[""]

    for path, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path, "r") as fp:
                song = fp.readlines()
                for i, voice in enumerate(song):
                    voice = voice.rstrip('\n')
                    tracks[i] += voice + " " + delimiter

    songs = "\n".join(tracks)
    with open(ONE_FILE_PATH, "w") as fp:
        for track in tracks:
            fp.write(f"{track}\n")

    return songs


def convert_songs_to_int(songs):
    """
        After all songs are in one file as strings with four lines (voices). This
        maps with the created ENCODER the strings to ints. It creates a numpy
        array (all tracks have the same length)

        PARAMETERS:
        --------------
        songs :     List of four strings. Every string represents one voice and all
                    pieces are seperated by a delimiter

        RETURNS:
        --------------
        np.array :  (4, number of all beats in total + SEQUENCE_LENGTH * Number of songs) - array
                    encoded in int format.

    """
    int_songs = [[] for _ in range(4)]

    with open(ENCODER_PATH, "r") as fp:
        encoder = json.load(fp)

    tracks = songs.split("\n")

    for i, track in enumerate(tracks):
        symbols = track.split()
        for symbol in symbols:
            int_songs[i].append(encoder[symbol])

    return np.asarray(int_songs)


def generate_training_sequences(sequence_length):
    """
        Create input and output data samples for training. Each sample is a sequence.

        PARAMETERS:
        --------------
        sequence_length :   Length for the sequences the LSTM should consider

        RETURN:
        --------------
        inputs :    (# of sequences, 4, sequence length, vocabulary size) numpy array
        targets:    (# of sequences, 4) numpy array

    """

    with open(ONE_FILE_PATH, "r") as fp:
        songs = fp.read()
    
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    num_sequences = np.shape(int_songs)[1] - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[:, i:i+sequence_length])
        targets.append(int_songs[:, i+sequence_length])

    vocabulary_size = len(set(list(int_songs.flatten())))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


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


if __name__=="__main__":
    print("Start preprocessing")
    #preprocess(BACH_CHORALES, ACCEPTABLE_DURATIONS, SAVEDIR)
    #songs = create_single_file(SAVEDIR, SEQUENCE_LENGTH)
    #create_dictionary(songs, ENCODER_PATH, DECODER_PATH)
    #int_songs = convert_songs_to_int(songs)
    #inp, out = generate_training_sequences(SEQUENCE_LENGTH)
    print("Finished")
