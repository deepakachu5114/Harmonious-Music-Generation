import music21 as m21
import numpy as np
import tensorflow.keras as keras
import json
from project_final.data import bbn


class MelodyGen:
    def __init__(
        self, model_path="/home/deepakachu/Desktop/DMS/project_final/model.h5"
    ):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(
                "/project_final/data/mapping.json", "r"
        ) as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * 64

    def generate(self, seed, num_steps, max_seq_len, temperature):
        # create seed with symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        # map seed to int
        # we are creating a seed and then mapping all the symbols
        # to the integers using the mapping dict
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # limit the seed to max_sequence_len
            seed = seed[-max_seq_len:]
            # one hot encode the seed
            one_hot_seed = keras.utils.to_categorical(
                seed, num_classes=len(self._mappings)
            )
            # the below line just adds an extra axis to convert the array into 3d
            one_hot_seed = one_hot_seed[np.newaxis, ...]

            # making a prediction
            probabilities = self.model.predict(one_hot_seed)[0]
            output = self._sample_with_temperature(probabilities, temperature)
            # update the seed so that we can feed it into network again
            seed.append(output)

            output_symbol = [k for k, v in self._mappings.items() if v == output][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                continue
            # if not, append the predicted note
            melody.append(output_symbol)
        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        """
        temp --> inf
        this will make the sampling much more random, not specific/rigid
        temp --> o
        this will make the sampling more specific/rigid
        """
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))  # returns a list of choice indices
        index = np.random.choice(choices, p=probabilities)  # we sample one randomly

        return index

    def save_melody(self, melody, mood):
        """Converts a melody into a MIDI file

        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """
        step_duration = 0

        if mood == "h":
            step_duration = 0.5
        elif mood == "s":
            step_duration = 1
        else:
            step_duration = 0.1
        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = (
                        step_duration * step_counter
                    )  # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(
                            int(start_symbol), quarterLength=quarter_length_duration
                        )

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(
            "midi",
            fp="/home/deepakachu/Desktop/DMS/project_final/generated_music/happy.mid",
        )
        # stream.write(format, file_name)


def lstm():
    import random

    # preprocess("/home/deepakachu/Desktop/DMS/project/content/essen/europa/deutschl/test")
    # songs = single_file("/home/deepakachu/Desktop/DMS/project/content/dataset", "/home/deepakachu/Desktop/DMS/project/content/dataset/single_file")
    # mapping(songs)
    # train()
    happy = [
        "69 _ 68 _ 72 _ 64 _ 62 _ 74 _ 60",
        "55 _ 57 _ 60 _ 62 _ 65 _ 67 _ 71",
        "72 _ 74 _ 76 _ 60 _ 64 _ 67 _ 69",
        "65 _ 67 _ 71 _ 74 _ 76 _ 60 _ 64",
        "67 _ 71 _ 74 _ 76 _ 60 _ 64 _ 55",
    ]
    sad = [
        "77 _ _ 76 _ _ 67 _ _ 65 _ _ 71 _ _ 55 _ _ 57",
        "69 _ _ 68 _ _ 72 _ _ 64 _ _ 62 _ _ 74 _ _ 60",
        "64 _ _ 55 _ _ 69 _ _ 72 _ _ 81 _ _ 76 _ _ 60",
        "60 _ _ 65 _ _ 57 _ _ 68 _ _ 74 _ _ 72 _ _ 81",
        "76 _ _ 67 _ _ 60 _ _ 65 _ _ 74 _ _ 72 _ _ 81",
    ]
    angry = [
        "77 _ 76 _ 67 _ 65 _ 71  _ 55 _ 57",
        "69 _ 68 _ 72 _ 64 _ 62 _ 74 _ 60",
        "64 _ 55 _ 69 _ 72 _ 81 _ 76  _ 60",
        "60 _ 65 _ 57 _ 68 _ 74 _ 72 _ 81",
        "76 _ 67 _ 60 _ 65 _ 74 _ 72 _ 81",
    ]
    mood = input("Enter mood: ")
    mg = MelodyGen()
    # seed = ""
    if mood == "s":
        seed = random.choice(sad)
    elif mood == "h":
        seed = random.choice(happy)
    else:
        seed = random.choice(angry)
    melody = mg.generate(seed, 500, 64, 0)
    mg.save_melody(melody, mood)


if __name__ == "__main__":
    mode = input("What method do you want to generate music by? \n1 for RNN-LSTM\n2 for BBNs ")
    if int(mode) == 1:
        lstm()
    elif int(mode) == 2:
        bbn.main()
    else:
        print("Error")
