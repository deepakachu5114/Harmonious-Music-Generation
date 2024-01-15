import music21 as m21
import numpy as np
import tensorflow.keras as keras
import json
import tkinter as tk
class MelodyGen:
    def __init__(
        self, model_path=r"/home/deepakachu/Desktop/DMS/project_final/data/model.h5"
    ):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(
                r"/project/Harmonious-Music-Generation/project_final/data/mapping.json", "r"
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
            fp=f"/home/deepakachu/Desktop/DMS/project_final/generated_music/LSTM_{mood}.mid",
        )
        # stream.write(format, file_name)


def lstm(mood):
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
    mg = MelodyGen()
    # seed = ""
    if mood == "s":
        seed = random.choice(sad)
    elif mood == "h":
        seed = random.choice(happy)
    else:
        seed = random.choice(angry)
    melody = mg.generate(seed, 500, 64, 10)
    mg.save_melody(melody, mood)
    play()

# Bayesic: This program takes in a number of parent and descendant nodes
# from the command line and, according to their assigned probabilities,
# samples the Bayesian network a fixed number of times, appending a note
# for each node that takes on the value 1 in the sample.

from music21 import *
import random
from tkinter import *
from tkinter.ttk import *
# each node will be on or off with prob p
import random
# import networkx as nx
# import matplotlib.pyplot as plt
import webbrowser
root = None
# Initialize a directed graph to represent the Bayesian network
def bay(emo):
    # G = nx.DiGraph()
    def bernoulli(p):
        event = random.random() < p
        if event:
            return 1
        else:
            return 0

    # begin interacting with user

    parents = []
    descendants = {}
    tiers = {}
    probs = {}
    freqs = {}

    inp = "Beginning..."
    print(inp)
#    emo = input("Enter emotion(h for happy, s for sad, a for angry): ")
    # get all parents

    """happy = [528, 261, 391, 138, 783, 146, 659, 277, 164, 293, 554, 46, 48, 54, 440, 184, 61, 65, 195, 69, 73, 329, 587, 82, 466, 219, 92, 97, 739, 233, 109, 493, 880, 369, 116, 246, 123]
    sad = [432, 97, 195, 164, 261, 41, 146, 246, 219, 130, 293, 329, 123, 109]
    angry = [261, 391, 523, 659, 32, 164, 41, 554, 43, 46, 48, 440, 184, 698, 65, 329, 587, 82, 466, 92, 97, 739, 493, 622, 369]"""

    happy = [261.93, 329.63, 392, 493.88, 587.33]
    sad = [220, 261.63, 329.63, 392, 493.88]
    angry = [311.13, 349.23, 466.16, 493.88, 523.23, 739.99]
    print("working")
    number_of_parents = 5
    print(number_of_parents)
    prob = (random.uniform(20, 40)) / 100

    while True:
        number_of_parents -= 1
        parent = f"{emo}-{number_of_parents +1}"
        probs[parent] = prob
        parents.append(parent)
        if number_of_parents == 0:
            break

    tiers[0] = parents

    # get all descendants

    cont = "yes"
    tierno = 1

    for desc, plist in descendants.items():
        if len(plist) == 1:
            prob0_label = desc + "_" + plist[0] + "0"
            prob1_label = desc + "_" + plist[0] + "1"

            prob0 = input("Enter the probability of " + desc + "|" + plist[0] + "= 0.")
            prob1 = input("Enter the probability of " + desc + "|" + plist[0] + "= 1.")

            probs[prob0_label] = float(prob0)
            probs[prob1_label] = float(prob1)

        else:
            prob00_label = desc + "_" + plist[0] + "0" + plist[1] + "0"
            prob01_label = desc + "_" + plist[0] + "0" + plist[1] + "1"
            prob10_label = desc + "_" + plist[0] + "1" + plist[1] + "0"
            prob11_label = desc + "_" + plist[0] + "1" + plist[1] + "1"

            prob00 = input(
                "Enter the probability of "
                + desc
                + "|"
                + plist[0]
                + "= 0,"
                + plist[1]
                + "= 0."
            )
            prob01 = input(
                "Enter the probability of "
                + desc
                + "|"
                + plist[0]
                + "= 0,"
                + plist[1]
                + "= 1."
            )
            prob10 = input(
                "Enter the probability of "
                + desc
                + "|"
                + plist[0]
                + "= 1,"
                + plist[1]
                + "= 0."
            )
            prob11 = input(
                "Enter the probability of "
                + desc
                + "|"
                + plist[0]
                + "= 1,"
                + plist[1]
                + "= 1."
            )

            probs[prob00_label] = float(prob00)
            probs[prob01_label] = float(prob01)
            probs[prob10_label] = float(prob10)
            probs[prob11_label] = float(prob11)

    # assign frequencies to each parent

    for parent in parents:
        if emo == "h":
            val = random.choice(happy)
            freqs[parent] = val
            happy.remove(val)
        elif emo == "s":
            val = random.choice(sad)
            freqs[parent] = val
            sad.remove(val)
        elif emo == "a":
            val = random.choice(angry)
            freqs[parent] = val
            angry.remove(val)

    ngens = 0
    if emo == "h":
        ngens = 500
    elif emo == "a":
        ngens = 800
    else:
        ngens = 200

    # # start making music!
    # for parent in parents:
    #     # G.add_node(parent)
    # for desc, plist in descendants.items():
    #     for parent in plist:
    #         G.add_edge(parent, desc)

    # Draw the Bayesian network
    # pos = nx.spring_layout(G)
    # nx.draw(
    #     G,
    #     pos,
    #     with_labels=True,
    #     node_size=1000,
    #     node_color="lightblue",
    #     font_size=10,
    #     font_color="black",
    #     font_weight="bold",
    # )
    # plt.title("Bayesian Network")
    # plt.show()

    waterfalls = stream.Stream()
    amp = 90
    amp_mod = 0.9  # 90% of volume for each tier
    w0 = 0.3  # if parent state 0, this much of it goes into average
    w1 = 1
    nodes = parents + list(descendants.keys())
    nodestates = {}

    for i in range(int(ngens)):
        if i > 0:
            r = note.Rest()  # a rest between each generation
            if emo == "a":
                r.duration.quarterLength = 0.1
            elif emo == "h":
                r.duration.quarterLength = 0.25
            else:
                r.duration.quarterLength = 0.4
            waterfalls.append(r)

        # sampling time!
        print("Sample ", i)

        for parent in parents:
            prob = probs[parent]
            state = bernoulli(prob)
            nodestates[parent] = state
            if state != 0:  # if parent on
                f = note.Note()
                f.pitch.frequency = float(freqs[parent])
                f.volume.velocity = amp
                if emo == "h":
                    f.duration.quarterLength = 0.25
                elif emo == "a":
                    f.duration.quarterLength = 0.2
                else:
                    f.duration.quarterLength = 1
                waterfalls.append(f)

        for tierno, tiernodes in tiers.items():
            if tierno == 0:
                continue

            for node in tiernodes:
                node_ps = descendants[node]  # all parents this node is conditioned on
                cond = ""
                nodef = 0

                for p in node_ps:
                    cond += p + str(nodestates[p])
                    if nodestates[p] == 0:
                        weight = w0
                    else:
                        weight = w1

                    nodef += float(freqs[p]) * weight

                nodef /= len(node_ps)  # weighted average
                label = node + "_" + cond

                prob_node = probs[label]
                freqs[node] = nodef
                state = bernoulli(prob_node)
                nodestates[node] = state

                if state != 0:
                    f = note.Note()
                    f.pitch.frequency = nodef
                    f.volume.velocity = amp * amp_mod**tierno
                    if emo == "h":
                        f.duration.quarterLength = 0.25
                    elif emo == "a":
                        f.duration.quarterLength = 0.2
                    else:
                        f.duration.quarterLength = 1
                    waterfalls.append(f)

        # nodestates is the whole sample now!
        print(nodestates)

    filename = (
        str(len(parents))
        + "_"
        + str(ngens)
        + "_"
        + str(len(list(descendants.keys())))
        + ".mid"
    )
    fp = f"/home/deepakachu/Desktop/DMS/project_final/generated_music/BBN_{emo}.mid"
    waterfalls.write("midi",fp,)
    play()
    popUp(fp)

#method to select the mood of BBN
def selectionBBN():
    global root
    lstmButton["state"]  = DISABLED
    bbnButton["state"] = DISABLED
    happyButton = tk.Button(root, text="happy", command=lambda: bay('h'), width=30, height=2)
    happyButton.pack()
    happyButton.place(x=390,y=300)
    sadButton = tk.Button(root, text="sad", command = lambda: bay('s'), width=30, height=2)
    sadButton.pack()
    sadButton.place(x=390,y=360)
    angryButton = tk.Button(root, text="angry", command=lambda: bay('a'), width=30, height=2)
    angryButton.pack()
    angryButton.place(x=390,y=420)
    tk.mainloop()
#method to select the mood of LSTM
def selectionLSTM():
    global root
    lstmButton["state"] = DISABLED
    bbnButton["state"] = DISABLED
    happyButton = tk.Button(root, text="happy", command=lambda:lstm('h'), width=30, height=2)
    happyButton.pack()
    happyButton.place(x=390,y=300)
    sadButton = tk.Button(root, text="sad", command=lambda: lstm('s') , width=30, height=2)
    sadButton.pack()
    sadButton.place(x=390,y=360)
    angryButton = tk.Button(root, text="angry", command=lambda: lstm('a'), width=30, height=2)
    angryButton.pack()
    angryButton.place(x=390,y=420)
    tk.mainloop()
def popUp(fileName):
    root.title("file saved")
    label = tk.Label(root,text = "file saved as"+fileName)
    label.pack()
    root.mainloop()

def play():
    url = "https://cifkao.github.io/html-midi-player/"
    webbrowser.open(url)
#these are global so we can DISABLE them in the method they call
lstmButton = None
bbnButton = None
def main():
    #this pseudo main method is necessary so we can use global variables which allow us to work in one
    #single window. we couldve called it something else like "driver" but ehhh
    global root,lstmButton,bbnButton
    root = tk.Tk()
    root.title("Music Generator")
    root.geometry("1000x639")
    img = PhotoImage(file =r'/project/Harmonious-Music-Generation/project_final/data/boombox.png', master=root)
    imgLabel = Label(root,image = img)
    imgLabel.place(x=0,y=0)
    lstmButton = tk.Button(root, text="RNN-LSTM", command=selectionLSTM , width=20, height=4)
    lstmButton.pack(pady=10)
    lstmButton.place(x=200, y=150)
#    lstmButton.place(x=180,y=360)
    bbnButton = tk.Button(root, text="BBN", command=selectionBBN, width=20, height=4)
    bbnButton.pack(pady=10)
    bbnButton.place(x=650, y=150)
    frame = Frame(root)
    frame.pack()
#    bbnButton.place(x=540,y=360)
    tk.mainloop()

if __name__ == "__main__":
    #real main method that calls pseudo main method
    main()